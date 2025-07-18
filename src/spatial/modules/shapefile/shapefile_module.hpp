#pragma once

namespace duckdb {

class ExtensionLoader;

void RegisterShapefileModule(ExtensionLoader &loader);

} // namespace duckdb