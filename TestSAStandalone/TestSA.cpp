#include <SuperNova>
#include <sycl/sycl.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>

using namespace BidirectionalInMemGraph;

// Defined by SuperNova's compiled core (core/src/SuperNova.cpp).
// Calling it proves this executable is linked against atomiccim_core,
// rather than only compiling SuperNova's headers.
extern "C" void atomiccim_core_compile_marker();

int main()
{
#ifndef SUPERNOVA_ENABLE_ADAPTIVECPP
    std::cerr << "FAIL: SUPERNOVA_ENABLE_ADAPTIVECPP is not defined.\n";
    return 1;
#endif

    // -------------------------------------------------------------------------
    // 1. Prove the separately built SuperNova core is linked.
    // -------------------------------------------------------------------------
    atomiccim_core_compile_marker();

    // Exercise a real SuperNova public utility from the architecture.
    constexpr std::uint32_t generation =
        HandleOfAPCStatic::FIRST_GENERATION;

    static_assert(
        HandleOfAPCStatic::IsGenerationValid(generation),
        "SuperNova generation invariant failed."
    );

    if (!HandleOfAPCStatic::IsGenerationValid(generation))
    {
        std::cerr << "FAIL: SuperNova generation validation failed.\n";
        return 2;
    }

    // -------------------------------------------------------------------------
    // 2. Run a real AdaptiveCpp/SYCL kernel.
    //
    //    expected[i] = input[i] * 2 + 1
    // -------------------------------------------------------------------------
    constexpr std::size_t N = 16;

    std::array<float, N> input{};
    std::array<float, N> output{};

    for (std::size_t i = 0; i < N; ++i)
    {
        input[i] = static_cast<float>(i);
        output[i] = 0.0f;
    }

    try
    {
        sycl::queue queue{sycl::default_selector_v};

        std::cout
            << "AdaptiveCpp device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << '\n';

        {
            sycl::buffer<float, 1> input_buffer{
                input.data(),
                sycl::range<1>{N}
            };

            sycl::buffer<float, 1> output_buffer{
                output.data(),
                sycl::range<1>{N}
            };

            queue.submit([&](sycl::handler& cgh)
            {
                auto in = input_buffer.get_access<sycl::access::mode::read>(cgh);
                auto out =
                    output_buffer.get_access<sycl::access::mode::write>(cgh);

                cgh.parallel_for(
                    sycl::range<1>{N},
                    [=](sycl::id<1> index)
                    {
                        out[index] = in[index] * 2.0f + 1.0f;
                    }
                );
            });

            queue.wait_and_throw();
        } // buffer destruction synchronizes/copies output back to host
    }
    catch (const sycl::exception& error)
    {
        std::cerr
            << "FAIL: AdaptiveCpp/SYCL exception: "
            << error.what()
            << '\n';
        return 3;
    }

    // -------------------------------------------------------------------------
    // 3. Numerical verification on the host.
    // -------------------------------------------------------------------------
    for (std::size_t i = 0; i < N; ++i)
    {
        const float expected = input[i] * 2.0f + 1.0f;

        if (std::fabs(output[i] - expected) > 1.0e-6f)
        {
            std::cerr
                << "FAIL: index=" << i
                << " expected=" << expected
                << " actual=" << output[i]
                << '\n';
            return 4;
        }
    }

    std::cout
        << "SuperNova core link: PASS\n"
        << "SuperNova generation API: PASS\n"
        << "AdaptiveCpp SYCL kernel: PASS\n"
        << "SuperNova + AdaptiveCpp integration: PASS\n";

    return 0;
}
