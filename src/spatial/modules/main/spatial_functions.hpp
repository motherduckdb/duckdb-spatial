#pragma once

#include "duckdb/common/typedefs.hpp"

namespace duckdb {

class ExtensionLoader;

void RegisterSpatialScalarFunctions(ExtensionLoader &loader);
void RegisterSpatialAggregateFunctions(ExtensionLoader &loader);
void RegisterSpatialCastFunctions(ExtensionLoader &loader);
void RegisterSpatialTableFunctions(ExtensionLoader &loader);

// TODO: Move these
class Vector;
struct CoreVectorOperations {
public:
	static void Point2DToVarchar(Vector &source, Vector &result, idx_t count);
	static void LineString2DToVarchar(Vector &source, Vector &result, idx_t count);
	static void Polygon2DToVarchar(Vector &source, Vector &result, idx_t count);
	static void Box2DToVarchar(Vector &source, Vector &result, idx_t count);
	static void GeometryToVarchar(Vector &source, Vector &result, idx_t count);
};

} // namespace duckdb