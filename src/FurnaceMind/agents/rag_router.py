def route_query(query: str):

    q = query.lower()

    # Data requests
    if any(k in q for k in [
        "trend", "plot", "last 8 hours",
        "temperature", "eta co",
        "fuel rate", "pressure"
    ]):
        return "influx"

    # Shift intelligence
    if any(k in q for k in [
        "shift", "fsi", "stability",
        "recurring", "influence", "anomaly"
    ]):
        return "shift"

    return "knowledge"