#include <gtest/gtest.h>

#include "dynamic_connection.hpp"
#include "reliable_connection.hpp"
#include "mpi.h"

// requires defined use_hdp_map to work

// test with different use_ib_hca
// test with different heap size
// test with different sleep
// test with different sq_size

TEST(DynamicConnect, ToNothing)
{
    DynamicConnection connect;
    connect.construct_init(1);
}

// test with different num_dcis
// test with different num_dcts

TEST(ReliableConnect, ToNothing)
{
    ReliableConnection connect;
    connect.construct_init(1);
}

int
main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
