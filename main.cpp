#include <iostream>
#include <string>
#include <string_view>

#include "LiveGraph/bind/livegraph.hpp"

int main()
{
    try
    {
        // LiveGraph supplies defaults for paths, block size and max vertex ID.
        lg::Graph graph{};

        constexpr lg::label_t EDGE_LABEL = 1;

        // --------------------------------------------------------
        // WRITE TRANSACTION
        // --------------------------------------------------------

        auto write_txn = graph.begin_transaction();

        const lg::vertex_t parent = write_txn.new_vertex();
        const lg::vertex_t child  = write_txn.new_vertex();

        write_txn.put_vertex(parent, "parent");
        write_txn.put_vertex(child,  "child");

        write_txn.put_edge(
            parent,
            EDGE_LABEL,
            child,
            "parent->child"
        );

        const lg::timestamp_t commit_timestamp = write_txn.commit();

        std::cout
            << "Write committed.\n"
            << "  parent = " << parent << '\n'
            << "  child  = " << child << '\n'
            << "  commit = " << commit_timestamp << "\n\n";

        // --------------------------------------------------------
        // READ-ONLY TRANSACTION
        // --------------------------------------------------------

        auto read_txn = graph.begin_read_only_transaction();

        const std::string_view parent_data =
            read_txn.get_vertex(parent);

        const std::string_view child_data =
            read_txn.get_vertex(child);

        const std::string_view edge_data =
            read_txn.get_edge(parent, EDGE_LABEL, child);

        std::cout
            << "Read transaction:\n"
            << "  parent data = " << parent_data << '\n'
            << "  child data  = " << child_data << '\n'
            << "  edge data   = " << edge_data << "\n\n";

        // --------------------------------------------------------
        // ADJACENCY SCAN
        // --------------------------------------------------------

        std::cout << "Outgoing edges from parent:\n";

        auto iterator =
            read_txn.get_edges(parent, EDGE_LABEL);

        while (iterator.valid())
        {
            std::cout
                << "  parent -> "
                << iterator.dst_id()
                << "   data = "
                << iterator.edge_data()
                << '\n';

            iterator.next();
        }

        read_txn.commit();

        std::cout << "\nLiveGraph smoke test PASSED.\n";

        return 0;
    }
    catch (const lg::Transaction::RollbackExcept& error)
    {
        std::cerr
            << "LiveGraph transaction rollback: "
            << error.what()
            << '\n';

        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "Error: "
            << error.what()
            << '\n';

        return 1;
    }
}