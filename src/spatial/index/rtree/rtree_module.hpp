#pragma once

namespace duckdb {

class DatabaseInstance;
class ExtensionLoader;

struct RTreeModule {
	static void RegisterIndex(DatabaseInstance &db);
	static void RegisterIndexScan(ExtensionLoader &loader);
	static void RegisterIndexPlanScan(DatabaseInstance &db);
	static void RegisterIndexPragmas(ExtensionLoader &loader);
};

} // namespace duckdb