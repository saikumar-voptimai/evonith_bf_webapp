import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from uuid import uuid4
from dotenv import load_dotenv
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"



# Environment setup
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = "evonith-anomaly-reports"
EMBEDDING_DIM = 384


# Vector Store Class
class VectorStore:

    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index_exists()
        self.index = self.pc.Index(INDEX_NAME)


    def _ensure_index_exists(self):
        existing = [i["name"] for i in self.pc.list_indexes()]
        if INDEX_NAME not in existing:
            print(f"Creating Pinecone index: {INDEX_NAME}")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            print(f"✅ Index '{INDEX_NAME}' already exists.")

   
    def text_to_embedding(self, text: str):
        return self.embeddings_model.embed_query(str(text))


    def parse_custom_timestamp(self, ts_str: str):
        """
        Safely parse timestamps like '2025-10-23T11-56-02' or '2025-10-23 T 11-56-02'
        into timezone-aware datetime (Asia/Kolkata).
        """
        if not ts_str or not isinstance(ts_str, str):
            return None

        ts_str = ts_str.strip().replace(" T ", "T").replace(" ", "")
        # Replace hyphens only in time part
        ts_str = re.sub(r"T(\d{2})-(\d{2})-(\d{2})", r"T\1:\2:\3", ts_str)

        try:
            dt = datetime.fromisoformat(ts_str)
        except ValueError:
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        return dt


    def store_embedding(self, text: str, metadata: dict):
        """
        Create embedding and store it in Pinecone.
        Retains timestamp and adds UTC + shift info for filtering.
        """

        if not isinstance(text, str):
            text = str(text)

        vector = self.text_to_embedding(text)

        # Normalize timestamp
        ts_str = metadata.get("timestamp", "")
        ts = self.parse_custom_timestamp(ts_str) or datetime.now(ZoneInfo("Asia/Kolkata"))

        # Compute UTC & shift info
        ts_utc = ts.astimezone(timezone.utc)
        ts_epoch = ts.timestamp()
        date_str = ts.strftime("%Y-%m-%d")
        hour = ts.hour
        shift = "A" if 0 <= hour < 8 else "B" if 8 <= hour < 16 else "C"

        # Prepare metadata
        meta = {
            "id": metadata.get("id", f"report-{uuid4().hex[:8]}"),
            "namespace": metadata.get("namespace", "default"),
            "source": metadata.get("source", "unknown"),
            "timestamp": ts.strftime("%Y-%m-%dT%H-%M-%S"),
            "timestamp_utc": ts_utc.isoformat(),
            "ts_epoch": ts_epoch,
            "date": date_str,
            "shift": shift,
        }

        # Save full text externally
        os.makedirs("stored_texts", exist_ok=True)
        safe_filename = f"{meta['id']}.txt".replace(":", "-").replace("+", "-")
        text_path = os.path.join("stored_texts", safe_filename)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        meta["text_path"] = text_path
        meta["preview"] = text[:500]

        # Validate size
        if len(json.dumps(meta)) > 40000:
            raise ValueError("Metadata exceeds 40KB limit — please trim preview or fields.")

        # Upsert to Pinecone
        self.index.upsert(vectors=[{
            "id": meta["id"],
            "values": vector,
            "metadata": meta,
        }])

        print(f"✅ Stored embedding with ID: {meta['id']} ({date_str} Shift {shift})")


    def similarity_search(self, query: str, k: int = 3):
        """
        Retrieve top-k similar records by query text.
        """
        query_vec = self.text_to_embedding(query)
        results = self.index.query(
            vector=query_vec,
            top_k=k,
            include_metadata=True,
        )

        formatted = []
        for match in results.matches:
            md = match.metadata or {}
            info = json.loads(md.get("info", "{}"))
            formatted.append({
                "id": match.id,
                "score": match.score,
                "text_snippet": md.get("text_ref", "")[:300],
                "metadata": info,
            })
        return formatted


    def fetch_latest_records(self):
        """
        Fetch the most recent *grouped* set of vectors that share the same time window.
        Handles pseudo-namespaces via metadata['namespace'].
        """

        all_records = []
        try:
            res = self.index.query(
                vector=[0] * EMBEDDING_DIM,
                top_k=100,
                include_metadata=True,
            )

            if not res or not getattr(res, "matches", []):
                print("⚠️ No records found.")
                return None

            for match in res.matches:
                meta = match.metadata or {}
                ns = meta.get("namespace", "default")
                ts = meta.get("timestamp", "")
                if ts:
                    all_records.append({
                        "namespace": ns,
                        "timestamp": ts,
                        "meta": meta,
                        "text_path": meta.get("text_path"),
                        "preview": meta.get("preview", "")
                    })
        except Exception as e:
            print(f"❌ Error fetching records: {e}")
            return None

        if not all_records:
            print("⚠️ No valid metadata found.")
            return None

        # Sort by timestamp descending
        all_records.sort(key=lambda x: x["timestamp"], reverse=True)

        # Group those within ±10 minutes of the latest
        latest_ts = all_records[0]["timestamp"]
        from datetime import datetime, timedelta, timezone

        latest_time = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
        time_window = timedelta(minutes=10)

        def in_window(ts):
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return abs((t - latest_time).total_seconds()) <= time_window.total_seconds()
            except Exception:
                return False

        grouped = [r for r in all_records if in_window(r["timestamp"])]

        # Pick only those from known namespaces
        valid_ns = {
            "EBF_df_summaries",
            "EBF_anomalies",
            "EBF_feedback",
            "EBF_operator_notes",
        }
        grouped = [r for r in grouped if r["namespace"] in valid_ns]

        # Load the actual text from stored files
        final = {}
        for r in grouped:
            ns = r["namespace"]
            path = r["text_path"]
            text = ""
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = r["preview"]
            final[ns] = {
                "timestamp": r["timestamp"],
                "text": text,
                "metadata": r["meta"],
            }

        return final
    

    def fetch_latest_record_by_namespace(self, namespace: str):
        """
        Fetch the most recent record from the vector DB for a given namespace.
        Returns a dict with {'timestamp', 'text', 'metadata'} or None.
        """
        try:
            res = self.index.query(
                vector=[0] * EMBEDDING_DIM,
                top_k=100,
                include_metadata=True,
            )

            if not res or not getattr(res, "matches", []):
                print("⚠️ No records found in vector DB.")
                return None

            # Filter only records from given namespace
            filtered_records = []
            for match in res.matches:
                meta = match.metadata or {}
                # fixed 'date' → 'timestamp'
                if meta.get("namespace") == namespace and "timestamp" in meta:
                    filtered_records.append({
                        "timestamp": meta["timestamp"],
                        "metadata": meta,
                        "text_path": meta.get("text_path"),
                        "preview": meta.get("preview", "")
                    })

            if not filtered_records:
                print(f"⚠️ No records found for namespace '{namespace}'.")
                return None

            # Sort descending by timestamp
            filtered_records.sort(key=lambda x: x["timestamp"], reverse=True)
            latest = filtered_records[0]

            # Load text from stored file if available
            text = ""
            if latest["text_path"] and os.path.exists(latest["text_path"]):
                with open(latest["text_path"], "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                text = latest["preview"]

            return {
                "timestamp": latest["timestamp"],
                "text": text,
                "metadata": latest["metadata"],
            }

        except Exception as e:
            print(f"❌ Error fetching latest record for namespace '{namespace}': {e}")
            return None
        
    
    def infer_shift(self, dt):
                h = dt.hour
                if 0 <= h < 8:
                    return "A"
                elif 8 <= h < 16:
                    return "B"
                else:
                    return "C"

    
    def fetch_records_by_date_shift(self, date_str: str, shift: str | None = None):
        """
        Fetch records for a specific date and shift based on custom timestamp format.
        """
        tz = ZoneInfo("Asia/Kolkata")

        shift_hours = {
            "A": ("00:00", "08:00"),
            "B": ("08:00", "16:00"),
            "C": ("16:00", "23:59"),
        }

        shift_list = [shift] if shift else ["A", "B", "C"]

        # Compute shift windows
        windows = []
        for sh in shift_list:
            if sh not in shift_hours:
                continue
            start_str, end_str = shift_hours[sh]
            start_dt = datetime.strptime(f"{date_str}T{start_str}", "%Y-%m-%dT%H:%M").replace(tzinfo=tz)
            end_dt = datetime.strptime(f"{date_str}T{end_str}", "%Y-%m-%dT%H:%M").replace(tzinfo=tz)
            windows.append((sh, start_dt, end_dt))

        # Query Pinecone (with metadata filter for date & shift)
        try:
            res = self.index.query(
                vector=[0]*EMBEDDING_DIM,
                top_k=1000,
                include_metadata=True,
                filter={"date": date_str} if not shift else {"date": date_str, "shift": shift},
            )
            matches = getattr(res, "matches", []) if res else []
        except Exception as e:
            print(f"❌ Vector DB query error: {e}")
            return None

        if not matches:
            print("⚠️ No records in vector DB.")
            return None

        valid_ns = {
            "EBF_df_summary",
            "EBF_anomaly_summary",
            "EBF_operator_feedback",
            "EBF_operator_notes",
        }

        final = {}
        for m in matches:
            meta = (getattr(m, "metadata", None) or {})
            ns = meta.get("namespace", "")
            ts_str = meta.get("timestamp", "")
            if ns not in valid_ns or not ts_str:
                continue

            ts_local = self.parse_custom_timestamp(ts_str)
            if not ts_local:
                ts_candidate = meta.get("id", "") or meta.get("text_path", "")
                match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", ts_candidate)
                if match:
                    ts_local = self.parse_custom_timestamp(match.group(1))

            if not ts_local:
                continue

            # Check shift window
            for sh, start_dt, end_dt in windows:
                if start_dt <= ts_local <= end_dt:
                    path = meta.get("text_path")
                    if path and not os.path.isabs(path):
                        path = os.path.join(os.getcwd(), path)
                    if path and os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            text = f.read()
                    else:
                        text = meta.get("preview", "")

                    key = f"{ns}_shift_{sh}"
                    final[key] = {
                        "timestamp": ts_local.isoformat(),
                        "shift": sh,
                        "text": text,
                        "metadata": meta,
                    }

        return final if final else None





# Global instance and helper functions

vs = VectorStore()

def summarize_report(report: str) -> str:
    return vs.summarize_report(report)

def df_summary(recent_df):
    return vs.df_summary(recent_df)

def text_to_embedding(text: str):
    return vs.text_to_embedding(text)

def store_embedding(text: str, metadata: dict):
    return vs.store_embedding(text, metadata)

def similarity_search(query: str, k: int = 3):
    return vs.similarity_search(query, k)

def fetch_latest_records():
    return vs.fetch_latest_records()

def fetch_latest_record_by_namespace(namespace: str):
    return vs.fetch_latest_record_by_namespace(namespace)

def fetch_records_by_date_shift(date_str: str, shift:str | None = None):
    return vs.fetch_records_by_date_shift(date_str, shift)

def infer_shift(dt):
    return vs.infer_shift(dt)

