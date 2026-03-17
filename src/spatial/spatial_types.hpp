#pragma once

#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/function/function.hpp"

namespace duckdb {

class ExtensionLoader;
struct LogicalType;
class ScalarFunction;
class AggregateFunction;
class ClientContext;
class Expression;

typedef unique_ptr<FunctionData> (*bind_scalar_function_t)(ClientContext &context, ScalarFunction &bound_function,
                                                           vector<unique_ptr<Expression>> &arguments);
typedef unique_ptr<FunctionData> (*bind_aggregate_function_t)(ClientContext &context, AggregateFunction &function,
                                                              vector<unique_ptr<Expression>> &arguments);

struct GeoTypes {
	static LogicalType POINT_2D();
	static LogicalType POINT_3D();
	static LogicalType POINT_4D();
	static LogicalType LINESTRING_2D();
	static LogicalType LINESTRING_3D();
	static LogicalType POLYGON_2D();
	static LogicalType POLYGON_3D();
	static LogicalType BOX_2D();
	static LogicalType BOX_2DF();

	static void Register(ExtensionLoader &loader);

	static LogicalType CreateEnumType(const string &name, const vector<string> &members);

	static unique_ptr<FunctionData> PropagateCRS(ClientContext &context, ScalarFunction &bound_function,
	                                             vector<unique_ptr<Expression>> &arguments);

	static unique_ptr<FunctionData> PropagateCRS(ClientContext &context, AggregateFunction &bound_function,
	                                             vector<unique_ptr<Expression>> &arguments);

	template <bind_aggregate_function_t BIND>
	static unique_ptr<FunctionData> PropagateCRS(ClientContext &context, AggregateFunction &bound_function,
	                                             vector<unique_ptr<Expression>> &arguments) {
		PropagateCRS(context, bound_function, arguments);
		return BIND(context, bound_function, arguments);
	}

	template <bind_scalar_function_t BIND>
	static unique_ptr<FunctionData> PropagateCRS(ClientContext &context, ScalarFunction &bound_function,
	                                             vector<unique_ptr<Expression>> &arguments) {
		PropagateCRS(context, bound_function, arguments);
		return BIND(context, bound_function, arguments);
	}
};

} // namespace duckdb
