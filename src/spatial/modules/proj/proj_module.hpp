#pragma once

namespace duckdb {

class ExtensionLoader;

void RegisterProjModule(ExtensionLoader &loader);

} // namespace duckdb