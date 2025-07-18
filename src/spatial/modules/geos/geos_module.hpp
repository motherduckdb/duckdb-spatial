#pragma once

namespace duckdb {

class ExtensionLoader;

void RegisterGEOSModule(ExtensionLoader &loader);

} // namespace duckdb