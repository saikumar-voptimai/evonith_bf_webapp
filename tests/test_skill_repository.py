from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.orm import sessionmaker

from furnace_data.relational.repositories import SkillRepository


def _exported_schema_skill_repository() -> SkillRepository:
    """Create a skills table matching the exported PostgreSQL schema."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    Table(
        "skills",
        metadata,
        Column("skill_id", String(64), primary_key=True),
        Column("name", String(256), nullable=False),
        Column("symbol", String(16)),
        Column("description", Text),
        Column("instruction", Text, nullable=False),
        Column("source_type", String(64), nullable=False),
        Column("qdrant_collection", String(128)),
        Column("is_active", Boolean, nullable=False),
        Column("created_by", String(128)),
        Column("metadata", JSON),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )
    metadata.create_all(engine)
    return SkillRepository(sessionmaker(bind=engine))


def test_skill_repository_matches_current_skills_table_with_symbol() -> None:
    """The repository should store symbols in the first-class SQL column."""
    repository = _exported_schema_skill_repository()

    created = repository.create_skill(
        name="Campaign Code Checker",
        symbol="CC",
        description="Answer questions about BF campaign codes.",
        instruction="Check uploaded knowledge context first.",
        source_type="uploaded",
        qdrant_collection="furnacemind_knowledge",
        is_active=True,
        created_by="900ef580-57a1-517e-8271-c384e3785057",
        metadata={"file_name": "dummy skill.md", "chunk_count": 1},
    )

    assert created.name == "Campaign Code Checker"
    assert created.symbol == "CC"
    assert created.description == "Answer questions about BF campaign codes."
    assert created.source_type == "uploaded"
    assert created.qdrant_collection == "furnacemind_knowledge"
    assert created.created_by == "900ef580-57a1-517e-8271-c384e3785057"
    assert created.metadata_json == {"file_name": "dummy skill.md", "chunk_count": 1}

    listed = repository.list_skills(active_only=True)

    assert [skill.skill_id for skill in listed] == [created.skill_id]
    assert listed[0].symbol == "CC"
    assert listed[0].metadata_json["file_name"] == "dummy skill.md"

    updated = repository.update_skill(
        skill_id=created.skill_id,
        name="Campaign Code Checker Updated",
        symbol="CU",
        qdrant_collection="furnacemind_skills",
        metadata={"file_name": "dummy skill.md", "chunk_count": 2},
    )

    assert updated is not None
    assert updated.name == "Campaign Code Checker Updated"
    assert updated.symbol == "CU"
    assert updated.qdrant_collection == "furnacemind_skills"
    assert updated.metadata_json["chunk_count"] == 2
