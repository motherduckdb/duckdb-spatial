#pragma once

#include "spatial/index/rtree/rtree_node.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {
class DuckTableEntry;
class Index;

// This is created by the optimizer rule
struct RTreeIndexScanBindData final : public TableFunctionData {
	explicit RTreeIndexScanBindData(DuckTableEntry &table, Index &index, const RTreeBounds &bbox,
	                                bool deferred_bounds = false)
	    : table(table), index(index), bbox(bbox), deferred_bounds(deferred_bounds) {
	}

	//! The table to scan
	DuckTableEntry &table;

	//! The index to use
	Index &index;

	//! The bounds to scan
	RTreeBounds bbox;

	//! If set, the bounds are not known at plan time, but only when the scan is initialized from a bounding-box filter
	//! pushed into the scan at runtime (e.g. by a spatial join build side).
	//! If no such filter arrives (or it is not selective enough), the scan falls back to a full table scan.
	bool deferred_bounds;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<RTreeIndexScanBindData>();
		return &other.table == &table;
	}
};

struct RTreeIndexScanFunction {
	static TableFunction GetFunction();
};

} // namespace duckdb
