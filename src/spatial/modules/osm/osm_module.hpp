#pragma once

namespace duckdb {

class ExtensionLoader;

void RegisterOSMModule(ExtensionLoader &loader);

} // namespace duckdb