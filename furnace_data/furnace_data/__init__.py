"""furnace_data — shared data-access core for the Evonith BF2 platform.

Provides a single entry point for all InfluxDB, PostgreSQL, and static-CSV
access used by the webapp and the furnace-data-service API.

Sub-packages
------------
config      load_config() — YAML config loader with env-var override
influx      BaseDataFetcher, query_builder, fetch_online_df, fetch_offline_data
domain      TemperatureDataFetcher, AverageHeatLoadDataFetcher
dataset     DatasetFetcher, DatasetService, StaticDatasetManager
relational  SQLAlchemy 2.0 engine/session helpers + ORM repositories
models      Shared Pydantic schemas (QueryType, OfflineReportType, …)
"""
