#pragma once

namespace duckdb {

class ExtensionLoader;

struct GeoArrow {
	static void Register(ExtensionLoader &loader);
};

} // namespace duckdb