# Nik's ML Library (NML)

The intention of this library was to build a super lightweight implementation of the most common machine learning data structures and algorithms with simple C++. This library is header-only and was designed to be extremely simple and easy to build/integrate into any project, including those targeting WebAssembly, Python, R, .NET, and Java.

## Roadmap

I am aiming to expand functionality, improve performance, and increase compatibility with various platforms and technologies. Here is what is coming:

- **PACMAP Implementation**: Finalize testing of the PACMAP implementation and develop a working usage example.

- **Enhanced Data Structures**: Introduce double precision support to `VectorSpan` and `MatrixSpan` to accommodate use cases where more precision is required.

- **XGBoost**: Add a native and simple implementation of XGBoost.

- **Parquet Support**: Implement support for reading and writing data in Parquet format to and from DataFrames.

- **Compiler Flags for Advanced Vectorization**:
  - **SSE and AVX**: Add compile flags for SSE and AVX to optimize performance on compatible CPUs, while maintaing compatability with simpler CPUs and Emscripten.
  - **CUDA**: Add CUDA compile flags to enable GPU acceleration where appropriate.

- **Threading Enhancements**: Develop a threading model that is compatible with Emscripten to allow performance improvements while ensuring compatibility and performance in web environments.

## Disclaimer

The tests for this library are not exhaustive and do not include fuzz testing. Please be aware that while I strive for robustness and reliability, the library may not handle all edge cases, and there are undiscovered bugs. Users are encouraged to thoroughly test the library in their environments, especially for critical applications.

This library is provided "as is," without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort or otherwise, arising from, out of, or in connection with the library or the use or other dealings in the library.

Community contributions to improve the library, including new algorithms, performance improvements, additional tests, and bug fixes are highly welcome and appreciated.

## External Dependencies
- The data structures and algorithms of this library rely on custom [data structures](library/nml/primitives) and do not use the STL. 
- The only external dependency is [Howard Hinnant's Date Library](https://github.com/HowardHinnant/date) found in [nml/external](library/nml/external).

## Memory
None of the [algorithms](library/nml/algorithms) in NML allocate any memory. They all depend on the struct RequestMemory to manage their memory needs externally, ensuring that memory management is both simple, efficient, and safe. 

Each algorithm in the library adheres to a standard API that includes:

```cpp
// Calculates and returns the amount of memory required for a given request.
static RequiredMemory required_memory(const Request& request) noexcept;

// Performs the computation using the pre-allocated memory.
static NMLResult<AlgorithmName> compute(const Request& request, RequestMemory& memory) noexcept;
```

To convert RequiredMemory into RequestMemory, one can simply use the following code:

```cpp
// Allocates and manages memory lifetime.
auto memory = MemoryOwner(required_memory.total_bytes());

// Transforms memory into RequestMemory
auto request_memory = memory.to_request_memory(required_memory);
```

This approach allows for pre-allocation of the necessary memory upfront into arenas based on the requirements of the algorithm, thereby minimizing dynamic memory allocation during runtime and enhancing simplicity, safety, and performance.

MemoryOwner and MatrixOwner are the two structs which manage their own memory via RAII. These are intended to be simple containers that produce MemorySpan and MatrixSpan, respectively, for performing operations without the overhead of additional memory management.

## Machine Learning Algorithms 
### [Principal Component Analysis](library/nml/algorithms/pca.h) (PCA)

```cpp
#include "library/nml/tests/datasets/test_data.h"

#include "library/nml/algorithms/pca.h"
#include "library/nml/primitives/file.h"
#include "library/nml/primitives/memory_owner.h"
#include "library/nml/primitives/matrix_owner.h"

using namespace nml;

int main()
{
    auto test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_VAL));

    auto test_span = test_matrix.to_span();

    auto pca_request = PCA::Request(test_span, test_span.column_ct);
    auto required_memory = PCA::required_memory(pca_request);

    auto memory = MemoryOwner(required_memory.total_bytes());
    auto request_memory = memory.to_request_memory(required_memory);

    auto pca_result = PCA::compute(pca_request, request_memory);

    if (pca_result.is_ok())
    {
        auto pca = pca_result.ok();
        pca.projection.print();
    }

    return 0;
}
```

### [Approximate Nearest Neighbors](library/nml/algorithms/approximate_nearest_neighbor.h) (ANN)

```cpp
#include "library/nml/tests/datasets/test_data.h"

#include "library/nml/primitives/file.h"
#include "library/nml/primitives/memory_owner.h"
#include "library/nml/primitives/matrix_owner.h"
#include "library/nml/algorithms/approximate_nearest_neighbor.h"

using namespace nml;

int main()
{
    unsigned thread_count = 1; // TODO multi threaded not yet implimented

    MatrixOwner test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_VAL));

    MatrixSpan test_matrix_span = test_matrix.to_span();

    auto build_request = ApproximateNearestNeighborIndex::BuildRequest(test_matrix_span, 10, thread_count, 42);

    auto required_build_memory = ApproximateNearestNeighborIndex::build_required_memory(build_request);

    auto memory = MemoryOwner(required_build_memory.total_bytes());

    auto build_request_memory = RequestMemory
    {
        .result_memory = memory.to_memory_span(0, required_build_memory.result_required_bytes),
        .working_memory = memory.to_memory_span(required_build_memory.result_required_bytes),
    };

    auto ann_index_result = ApproximateNearestNeighborIndex::build(build_request, build_request_memory);

    if (ann_index_result.is_err())
    {
        std::cout << "Failed to build index." << std::endl;
        return (int)ann_index_result.err();
    }

    auto ann_index = ann_index_result.ok();

    auto bulk_find_request = ApproximateNearestNeighborIndex::BulkFindRequest(12, 0, thread_count);

    auto bulk_find_required_memory = ann_index.bulk_find_required_memory(bulk_find_request);

    auto bulk_find_memory = MemoryOwner(bulk_find_required_memory.total_bytes());

    auto find_request_memory = RequestMemory::from_required_unsafe(bulk_find_required_memory, bulk_find_memory.to_memory_span());

    auto bulk_nearest_neighbors_result = ann_index.bulk_find_nearest_neighbors(bulk_find_request, find_request_memory);

    if (bulk_nearest_neighbors_result.is_err())
    {
        std::cout << "Failed to bulk find." << std::endl;
        return (int)bulk_nearest_neighbors_result.err();
    }

    auto nearest_neighbors = bulk_nearest_neighbors_result.ok();

    nearest_neighbors.print();

    return 0;
}
```

### [Eigen Solvers](library/nml/algorithms/eigen_solver.h)

```cpp
#include "library/nml/primitives/file.h"
#include "library/nml/algorithms/eigen_solver.h"
#include "library/nml/primitives/memory_owner.h"

using namespace nml;

const float zero_threshold = 1e-06;

int main()
{
    MatrixOwner test_matrix = MatrixOwner(4, 4, {
         1.00671141, -0.11835884,  0.87760447,  0.82343066,
        -0.11835884,  1.00671141, -0.43131554, -0.36858315,
         0.87760447, -0.43131554,  1.00671141,  0.96932762,
         0.82343066, -0.36858315,  0.96932762,  1.00671141
    });

    MatrixOwner expected_matrix = MatrixOwner(4, 4, {
         0.521067, 0.3774190,  0.719563, -0.261292,
        -0.269344, 0.923296 , -0.244384,  0.123512,
         0.580413, 0.0244887, -0.142120,  0.801451,
         0.564857, 0.0669382, -0.634277, -0.523592
    });

    MatrixSpan test_span = test_matrix.to_span();
    MatrixSpan expected_span = expected_matrix.to_span();

    auto eigen_request = Eigen::Request(test_span);

    auto required_memory = Eigen::required_memory(eigen_request);
    auto memory = MemoryOwner(required_memory.total_bytes());
    auto request_memory = memory.to_request_memory(required_memory);

    {
        eigen_request.type = QRDecomposition::Type::GIVENS;
        auto eigen_result = Eigen::compute(eigen_request, request_memory);
        
        if (eigen_result.is_ok())
        {
            auto eigen = eigen_result.ok();
            eigen.print();
        }
    }

    {
        eigen_request.type = QRDecomposition::Type::HOUSEHOLDER;
        auto eigen_result = Eigen::compute(eigen_request, request_memory);
        
        if (eigen_result.is_ok())
        {
            auto eigen = eigen_result.ok();
            eigen.print();
        }
    }
    
    return 0;
}
```

### [Pairwise Controlled Manifold Approximation](library/nml/algorithms/pacmap.h) (PACMAP)

// TODO sample

### [Linear Regression](library/nml/algorithms/ordinary_least_squares.h) (OLS)

```cpp
#include "library/nml/primitives/file.h"
#include "library/nml/primitives/memory_owner.h"
#include "library/nml/algorithms/ordinary_least_squares.h"

using namespace nml;

const float zero_threshold = 1e-06;

int main()
{
    auto test_matrix = MatrixOwner(100'000, 4);

    auto test_span = test_matrix.to_span();

    test_span.fill_random_gaussian();

    for (unsigned row = 0; row < test_span.row_ct; ++row)
    {
        auto test_row = test_span[row];

        test_row[0] += 4 + 5 * test_row[1] + 7 * (test_row[2]) + 8 * (test_row[3]);
    }

    auto request = OLS::Request(test_span);

    auto required_memory = OLS::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto ols_result = OLS::compute(request, memory.to_request_memory(required_memory));

    if (ols_result.is_err())
    {
        std::cout << "Failed compute OLS." << std::endl;
        return (int)ols_result.err();
    }

    auto ols = ols_result.ok();

    ols.summary(test_span).print();

    return 0;
}
```

### [Logistic Regression](library/nml/algorithms/logistic_regression.h)

```cpp
#include "library/nml/tests/datasets/test_data.h"

#include "library/nml/primitives/file.h"
#include "library/nml/primitives/memory_owner.h"
#include "library/nml/primitives/matrix_owner.h"
#include "library/nml/algorithms/logistic_regression.h"

using namespace nml;

const float zero_threshold = 1e-06;

int main()
{
    float label = 2.0f;

    auto test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_LAB));

    auto test_span = test_matrix.to_span();

    auto request = LogisticRegression::Request(test_span);

    request.label = label;

    auto required_memory = LogisticRegression::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto result = LogisticRegression::compute(request, memory.to_request_memory(required_memory));

    if (result.is_err())
    {
        std::cout << "Failed compute Logistic Regression." << std::endl;
        return (int)result.err();
    }

    auto regression = result.ok();

    regression.summary(test_span).print();
}
```

## Data Transformation

### [Data Frame](library/nml/primitives/data_frame.h)

```cpp
#include "library/nml/tests/datasets/test_data.h"

#include "library/nml/primitives/data_frame.h"
#include "library/nml/primitives/memory_owner.h"

using namespace nml;

static void from_file()
{
    auto df = DataFrame::from_delimited(TestData::test_file_path(Dataset::IRIS_RAW));

    df.print(10);
}

static void basic_operations()
{
    auto df = DataFrame();

    auto ints = ResizableList<int64_t>();

    for (int i = 0; i < 10; ++i)
    {
        ints.add(i);
    }

    df.add_column(ints.to_span()).set_column_name("one");
    df.add_column(ints.to_span()).set_column_name("two");

    df.add_row({1, 2});

    df[0][1] = -9223372036854775807LL;
    df[1][1] = 9223372036854775807LL;

    for (int64_t i = 2; i < 10; ++i)
    {
        df[i][1] += -i * 10000000000000000;
    }

    auto memory = MemoryOwner(2 * sizeof(DataFrame::Value));

    df.add_column(DataFrame::Type::INT, "int column");

    for (int i = 0; i < 10; ++i)
    {
        df.add_row({i, i + 1, i + 3});
    }

    df.add_column(df.get_column_unsafe(1));
    auto float_col = df.add_column(DataFrame::Type::FLOAT, "floats", 0);

    float_col[0] = 1.1;

    auto copy = df.add_column(df.get_column_unsafe(0));

    copy.set_column_name("copy column");

    float_col = -11111111111111.1111111111111f;
    float_col += 1.1f;

    df[0][0] = 1 * 10000000000000000.0f;
    df[df.row_count() - 1][df.column_count() - 1] = 2.1f;

    auto str_col = df.add_column(DataFrame::Value("test string 1"), "dynamic");
    df.print(5);
    str_col = 100.001;
    df.print(5);
    str_col = "string 100.123456789012345";
    df.print(5);
    str_col = 1;
    df.print(5);
    str_col = DataFrame::Value::parse_datetime("1992-02-18");
    df.print(5);
}

int main()
{
    from_file();

    basic_operations();

    return 0;
}

```

### [Regular Expressions](library/nml/primitives/regular_expressions.h)

```cpp
#include "library/nml/primitives/regular_expressions.h"

using namespace nml;

int main()
{
    const char* string = "<img src=\"HTTPS://google.cOm/\"/> "
            "  <a href=\"http://en.wikipedia.org/wiki/Eigenvalue_algorithm\">Eigenvalues</a>";

    auto regex_result = Regex::compile(R"(((https?://)[^\s/'"<>]+/?[^\s'"<>]*))", Regex::Option::INSENSITIVE);

    if (regex_result.is_err())
    {
        std::cout << "Failed to parse." << std::endl;
        return (int)regex_result.err();
    }

    auto regex = regex_result.ok();

    auto matches = regex.matches(string);

    while (matches.has_next())
    {
        auto current = matches.next();

        printf("Found URL: %.*s\n", current.length, current.match);
    }

    return 0;
}
```

## Data Structures

### [Heap (Min and Max)](library/nml/primitives/heap.h)
#### MaxHeap
```cpp
#include "library/nml/primitives/heap.h"
#include "library/nml/primitives/memory_owner.h"

using namespace nml;

int main()
{
    const int queue_size = 20;

    auto memory = MemoryOwner(Heap<int>::required_bytes(queue_size));

    auto memory_span = memory.to_memory_span();

    auto heap_max = Heap<int, false>(memory_span);

    for (int i = 0; i < queue_size; ++i)
    {
        heap_max.push(i);
    }

    while (!heap_max.is_empty())
    {
        std::cout << heap_max.pop() << std::endl;
    }

    for (int i = queue_size; i > 0; --i)
    {
        heap_max.push(i);
    }

    while (!heap_max.is_empty())
    {
        std::cout << heap_max.pop() << std::endl;
    }

    return 0;
}
```
#### MinHeap
```cpp
#include "library/nml/primitives/heap.h"
#include "library/nml/primitives/memory_owner.h"

using namespace nml;

int main()
{
    const int queue_size = 20;

    auto memory = MemoryOwner(Heap<int>::required_bytes(queue_size));

    auto memory_span = memory.to_memory_span();

    auto heap_min = Heap<int, true>(memory_span);

    for (int i = 0; i < queue_size; ++i)
    {
        heap_min.push(i);
    }

    while (!heap_min.is_empty())
    {
        std::cout << heap_min.pop() << std::endl;
    }

    for (int i = queue_size; i > 0; --i)
    {
        heap_min.push(i);
    }

    while (!heap_min.is_empty())
    {
        std::cout << heap_min.pop() << std::endl;
    }

    return 0;
}
```
### [MinMaxHeap](library/nml/primitives/heap_min_max.h)
```cpp
#include "library/nml/primitives/heap_min_max.h"
#include "library/nml/primitives/memory_owner.h"

using namespace nml;

int main()
{
    unsigned queue_size = 20;

    auto memory = MemoryOwner(MinMaxHeap<int>::required_bytes(queue_size));

    auto memory_span = memory.to_memory_span();

    auto queue = MinMaxHeap<int>(memory_span);

    for (int i = 0; i < 1000; ++i)
    {
        queue.insert_min(i);
    }
    
    while (!queue.is_empty())
    {
        std::cout << queue.remove_min_unsafe() << std::endl;
    }

    for (int i = 0; i < 1000; ++i)
    {
        queue.insert_max(i);
    }
    
    while (!queue.is_empty())
    {
        std::cout << queue.remove_max_unsafe() << std::endl;
    }

    return 0;
}
```
### [HashSet](library/nml/primitives/hash.h)

```cpp
#include "library/nml/primitives/hash.h"

using namespace nml;

int main()
{
    char string[] = "string 1 | string 2 | string 3 | string 4 | string 5";

    auto set = HashSet<Span<char>>();

    for (unsigned i = 0; i < 10; ++i)
    {
        set.insert(Span<char>(string + 0, 8));
        set.insert(Span<char>(string + 11, 8));
        set.insert(Span<char>(string + 22, 8));
        set.insert(Span<char>(string + 33, 8));
        set.insert(Span<char>(string + 44, 8));
    }

    int ct = 0;

    auto iterator = set.to_iterator();

    while (iterator.has_next())
    {
        auto next = iterator.next();

        if (ct++ > 0) std::cout << ", ", next.print("", false);
        else next.print("", false);
    }

    return 0;
}
```
### [HashMap](library/nml/primitives/hash.h)

```cpp
#include "library/nml/primitives/hash.h"

using namespace nml;

int main()
{
    auto map = HashMap<int32_t, uint32_t>();

    for (int i = -10; i < 10; ++i)
    {
        map.insert(i, map.count() * 2);
        map.insert(i, map.count() * 2);
        map.insert(i, map.count() * 2);
        map.insert(i, map.count() * 2);
        map.insert(i, map.count() * 2);
    }

    for (int i = -100; i < 100; ++i)
    {
        auto value = *map.get_value(i);
        std::cout << value << std::endl;
    }

    return 0;
}
```

