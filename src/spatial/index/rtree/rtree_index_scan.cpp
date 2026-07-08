#include "spatial/index/rtree/rtree_module.hpp"
#include "spatial/index/rtree/rtree_index.hpp"
#include "spatial/index/rtree/rtree_index_scan.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/dependency_list.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/storage/data_table.hpp"

namespace duckdb {

BindInfo RTreeIndexScanBindInfo(const optional_ptr<FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<RTreeIndexScanBindData>();
	return BindInfo(bind_data.table);
}

//-------------------------------------------------------------------------
// Global State
//-------------------------------------------------------------------------
struct RTreeIndexScanGlobalState final : public GlobalTableFunctionState {
	//! The maximum number of threads for this scan
	idx_t max_threads = 1;

	//! The storage column ids to fetch
	vector<StorageIndex> column_ids;
	//! The types of all scanned columns, including filter columns that are removed afterwards
	vector<LogicalType> scanned_types;
	vector<idx_t> projection_ids;

	//! Lock protecting the shared state below
	mutex lock;
	//! The index scan state, shared by all threads
	unique_ptr<IndexScanState> index_state;
	//! Whether the index scan is exhausted
	bool index_exhausted = false;
	//! Whether a thread has been assigned to scan the transaction-local storage
	bool local_storage_scan_assigned = false;

	idx_t MaxThreads() const override {
		return max_threads;
	}
	bool CanRemoveFilterColumns() const {
		return !projection_ids.empty();
	}
};

static unique_ptr<GlobalTableFunctionState> RTreeIndexScanInitGlobal(ClientContext &context,
                                                                     TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<RTreeIndexScanBindData>();
	auto result = make_uniq<RTreeIndexScanGlobalState>();

	// While the index itself is scanned by one thread at a time, fetching the rows can happen in parallel
	result->max_threads = bind_data.table.GetStorage().MaxThreads(context);

	// Figure out the storage column ids
	result->column_ids.reserve(input.column_ids.size());
	for (auto &id : input.column_ids) {
		storage_t col_id = id;
		if (id != DConstants::INVALID_INDEX) {
			col_id = bind_data.table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->column_ids.emplace_back(col_id);
	}

	// Initialize the scan state for the index
	result->index_state = bind_data.index.Cast<RTreeIndex>().InitializeScan(bind_data.bbox);

	// Early out if there is nothing to project
	if (!input.CanRemoveFilterColumns()) {
		return std::move(result);
	}

	// We need this to project out what we scan from the underlying table.
	result->projection_ids = input.projection_ids;

	auto &duck_table = bind_data.table.Cast<DuckTableEntry>();
	const auto &columns = duck_table.GetColumns();
	for (const auto &col_idx : input.column_indexes) {
		if (col_idx.IsRowIdColumn()) {
			result->scanned_types.emplace_back(LogicalType::ROW_TYPE);
		} else {
			result->scanned_types.push_back(columns.GetColumn(col_idx.ToLogical()).Type());
		}
	}

	return std::move(result);
}

//-------------------------------------------------------------------------
// Local State
//-------------------------------------------------------------------------
struct RTreeIndexScanLocalState final : public LocalTableFunctionState {
	//! The row ids scanned from the index by this thread
	Vector row_ids = Vector(LogicalType::ROW_TYPE);
	//! The fetch state used to fetch rows from the main storage
	ColumnFetchState fetch_state;
	//! Scan state for the transaction-local storage
	TableScanState local_storage_state;
	vector<StorageIndex> column_ids;
	//! The DataChunk containing all read columns.
	//! This includes filter columns, which are immediately removed.
	DataChunk all_columns;
	//! Whether this thread is in charge of scanning the transaction-local storage
	bool in_charge_of_local_storage = false;
};

static unique_ptr<LocalTableFunctionState> RTreeIndexScanInitLocal(ExecutionContext &context,
                                                                   TableFunctionInitInput &input,
                                                                   GlobalTableFunctionState *global_state) {
	auto &bind_data = input.bind_data->Cast<RTreeIndexScanBindData>();
	auto &g_state = global_state->Cast<RTreeIndexScanGlobalState>();
	auto result = make_uniq<RTreeIndexScanLocalState>();

	// Setup the scan state for the local storage
	auto &local_storage = LocalStorage::Get(context.client, bind_data.table.catalog);
	result->column_ids = g_state.column_ids;
	result->local_storage_state.Initialize(result->column_ids, context.client, input.filters);
	local_storage.InitializeScan(bind_data.table.GetStorage(), result->local_storage_state.local_state, input.filters);

	if (g_state.CanRemoveFilterColumns()) {
		result->all_columns.Initialize(context.client, g_state.scanned_types);
	}
	return std::move(result);
}

//-------------------------------------------------------------------------
// Execute
//-------------------------------------------------------------------------
static void RTreeIndexScanExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

	auto &bind_data = data_p.bind_data->Cast<RTreeIndexScanBindData>();
	auto &g_state = data_p.global_state->Cast<RTreeIndexScanGlobalState>();
	auto &l_state = data_p.local_state->Cast<RTreeIndexScanLocalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);
	auto &index = bind_data.index.Cast<RTreeIndex>();

	enum class ExecutionPhase { NONE, STORAGE, LOCAL_STORAGE };

	// We might need to loop back if a fetched batch turns out to be empty
	while (true) {
		idx_t row_count = 0;
		auto phase = ExecutionPhase::NONE;
		{
			// Scan the index for the next batch of row ids while holding the lock
			lock_guard<mutex> guard(g_state.lock);
			if (!g_state.index_exhausted) {
				row_count = index.Scan(*g_state.index_state, l_state.row_ids);
				if (row_count != 0) {
					phase = ExecutionPhase::STORAGE;
				} else {
					g_state.index_exhausted = true;
				}
			}
			if (phase == ExecutionPhase::NONE) {
				// The index is exhausted: the first thread to get here is assigned to scan whatever is in the
				// transaction-local storage, all other threads are done.
				if (!g_state.local_storage_scan_assigned) {
					g_state.local_storage_scan_assigned = true;
					l_state.in_charge_of_local_storage = true;
				}
				if (l_state.in_charge_of_local_storage) {
					phase = ExecutionPhase::LOCAL_STORAGE;
				}
			}
		}

		switch (phase) {
		case ExecutionPhase::NONE: {
			// No more work to pick up
			return;
		}
		case ExecutionPhase::STORAGE: {
			// Fetch the data from the main storage given the row ids, in parallel, without holding the lock
			if (!g_state.CanRemoveFilterColumns()) {
				bind_data.table.GetStorage().Fetch(transaction, output, g_state.column_ids, l_state.row_ids, row_count,
				                                   l_state.fetch_state);
			} else {
				// We need to first fetch into our scan chunk, and then project out the result
				l_state.all_columns.Reset();
				bind_data.table.GetStorage().Fetch(transaction, l_state.all_columns, g_state.column_ids,
				                                   l_state.row_ids, row_count, l_state.fetch_state);
				output.ReferenceColumns(l_state.all_columns, g_state.projection_ids);
			}
			if (output.size() == 0) {
				if (data_p.results_execution_mode == AsyncResultsExecutionMode::TASK_EXECUTOR) {
					// We can avoid looping, and just return as appropriate
					data_p.async_result = AsyncResultType::HAVE_MORE_OUTPUT;
					return;
				}
				// The whole batch got filtered out (e.g. deleted rows), loop back and grab more work
				continue;
			}
			return;
		}
		case ExecutionPhase::LOCAL_STORAGE: {
			// Scan the transaction-local storage, sequentially, always on the same thread.
			// This won't be indexed, but at least we get the correct results.
			auto &local_storage = LocalStorage::Get(transaction);
			if (!g_state.CanRemoveFilterColumns()) {
				local_storage.Scan(l_state.local_storage_state.local_state, l_state.column_ids, output);
			} else {
				// We need to scan into our scan chunk, and then project out the result
				l_state.all_columns.Reset();
				local_storage.Scan(l_state.local_storage_state.local_state, l_state.column_ids, l_state.all_columns);
				output.ReferenceColumns(l_state.all_columns, g_state.projection_ids);
			}
			return;
		}
		}
	}
}

//-------------------------------------------------------------------------
// Statistics
//-------------------------------------------------------------------------
static unique_ptr<BaseStatistics> RTreeIndexScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                                                           column_t column_id) {
	auto &bind_data = bind_data_p->Cast<RTreeIndexScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	if (local_storage.Find(bind_data.table.GetStorage())) {
		// we don't emit any statistics for tables that have outstanding transaction-local data
		return nullptr;
	}
	return bind_data.table.GetStatistics(context, column_id);
}

//-------------------------------------------------------------------------
// Dependency
//-------------------------------------------------------------------------
void RTreeIndexScanDependency(LogicalDependencyList &entries, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<RTreeIndexScanBindData>();
	entries.AddDependency(bind_data.table);

	// TODO: Add dependency to index here?
}

//-------------------------------------------------------------------------
// Cardinality
//-------------------------------------------------------------------------
unique_ptr<NodeStatistics> RTreeIndexScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<RTreeIndexScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	const auto &storage = bind_data.table.GetStorage();
	idx_t table_rows = storage.GetTotalRows();
	idx_t estimated_cardinality = table_rows + local_storage.AddedRows(bind_data.table.GetStorage());
	return make_uniq<NodeStatistics>(table_rows, estimated_cardinality);
}

//-------------------------------------------------------------------------
// Virtual Columns
//-------------------------------------------------------------------------
static virtual_column_map_t RTreeIndexScanGetVirtualColumns(ClientContext &context,
                                                            optional_ptr<FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<RTreeIndexScanBindData>();
	return bind_data.table.GetVirtualColumns();
}

static vector<column_t> RTreeIndexScanGetRowIdColumns(ClientContext &context, optional_ptr<FunctionData> bind_data) {
	vector<column_t> result;
	result.emplace_back(COLUMN_IDENTIFIER_ROW_ID);
	return result;
}

//-------------------------------------------------------------------------
// ToString
//-------------------------------------------------------------------------
static InsertionOrderPreservingMap<string> RTreeIndexScanToString(TableFunctionToStringInput &input) {
	D_ASSERT(input.bind_data);
	InsertionOrderPreservingMap<string> result;
	auto &bind_data = input.bind_data->Cast<RTreeIndexScanBindData>();
	result["Table"] = bind_data.table.name;
	result["Index"] = bind_data.index.GetIndexName();
	return result;
}

//-------------------------------------------------------------------------
// De/Serialize
//-------------------------------------------------------------------------
static void RTreeScanSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                               const TableFunction &function) {
	auto &bind_data = bind_data_p->Cast<RTreeIndexScanBindData>();
	serializer.WriteProperty(100, "catalog", bind_data.table.schema.catalog.GetName());
	serializer.WriteProperty(101, "schema", bind_data.table.schema.name);
	serializer.WriteProperty(102, "table", bind_data.table.name);
	serializer.WriteProperty(103, "index_name", bind_data.index.GetIndexName());

	serializer.WriteObject(104, "bbox", [&](Serializer &ser) {
		ser.WriteProperty<float>(10, "min_x", bind_data.bbox.min.x);
		ser.WriteProperty<float>(11, "min_y", bind_data.bbox.min.y);
		ser.WriteProperty<float>(20, "max_x", bind_data.bbox.max.x);
		ser.WriteProperty<float>(21, "max_y", bind_data.bbox.max.y);
	});
}

static unique_ptr<FunctionData> RTreeScanDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto &context = deserializer.Get<ClientContext &>();

	const auto catalog = deserializer.ReadProperty<string>(100, "catalog");
	const auto schema = deserializer.ReadProperty<string>(101, "schema");
	const auto table = deserializer.ReadProperty<string>(102, "table");
	auto &catalog_entry = Catalog::GetEntry<TableCatalogEntry>(context, catalog, schema, table);
	if (catalog_entry.type != CatalogType::TABLE_ENTRY) {
		throw SerializationException("Cant find table for %s.%s", schema, table);
	}

	// Now also lookup the index by name
	const auto index_name = deserializer.ReadProperty<string>(103, "index_name");
	RTreeBounds bbox;
	deserializer.ReadObject(104, "bbox", [&](Deserializer &ser) {
		bbox.min.x = ser.ReadProperty<float>(10, "min_x");
		bbox.min.y = ser.ReadProperty<float>(11, "min_y");
		bbox.max.x = ser.ReadProperty<float>(20, "max_x");
		bbox.max.y = ser.ReadProperty<float>(21, "max_y");
	});

	auto &duck_table = catalog_entry.Cast<DuckTableEntry>();
	auto &table_info = *catalog_entry.GetStorage().GetDataTableInfo();

	unique_ptr<RTreeIndexScanBindData> result = nullptr;

	table_info.BindIndexes(context, RTreeIndex::TYPE_NAME);
	for (auto &index : table_info.GetIndexes().Indexes()) {
		if (!index.IsBound() || RTreeIndex::TYPE_NAME != index.GetIndexType()) {
			continue;
		}
		auto &index_entry = index.Cast<RTreeIndex>();
		if (index_entry.GetIndexName() == index_name) {
			result = make_uniq<RTreeIndexScanBindData>(duck_table, index_entry, bbox);
			break;
		}
	};

	if (!result) {
		throw SerializationException("Could not find index %s on table %s.%s", index_name, schema, table);
	}
	return std::move(result);
}

//-------------------------------------------------------------------------
// Get Function
//-------------------------------------------------------------------------
TableFunction RTreeIndexScanFunction::GetFunction() {
	TableFunction func("rtree_index_scan", {}, RTreeIndexScanExecute);
	func.init_local = RTreeIndexScanInitLocal;
	func.init_global = RTreeIndexScanInitGlobal;
	func.statistics = RTreeIndexScanStatistics;
	func.dependency = RTreeIndexScanDependency;
	func.cardinality = RTreeIndexScanCardinality;
	func.pushdown_complex_filter = nullptr;
	func.to_string = RTreeIndexScanToString;
	func.table_scan_progress = nullptr;
	func.projection_pushdown = true;
	func.filter_pushdown = false;
	func.get_bind_info = RTreeIndexScanBindInfo;
	func.serialize = RTreeScanSerialize;
	func.deserialize = RTreeScanDeserialize;
	func.get_virtual_columns = RTreeIndexScanGetVirtualColumns;
	func.get_row_id_columns = RTreeIndexScanGetRowIdColumns;

	return func;
}

//-------------------------------------------------------------------------
// Register
//-------------------------------------------------------------------------
void RTreeModule::RegisterIndexScan(ExtensionLoader &loader) {
	loader.RegisterFunction(RTreeIndexScanFunction::GetFunction());
}

} // namespace duckdb
