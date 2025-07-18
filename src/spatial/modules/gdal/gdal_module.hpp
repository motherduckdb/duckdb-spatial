#pragma once

namespace duckdb {

class ExtensionLoader;

void RegisterGDALModule(ExtensionLoader &db);

} // namespace duckdb