#include <functional>

template <typename CoordinatePrecision, int dimension>
using Point = std::array<CoordinatePrecision, dimension>;

template <typename CoordinatePrecision, typename CoefficientPrecision, int dimension>
using basis_function_type =
    const std::function<
        CoefficientPrecision(
            std::vector<Point<CoordinatePrecision, dimension>>,
            int,
            Point<CoordinatePrecision, dimension>)>;

template <
    typename CoordinatePrecision,
    typename CoefficientPrecision,
    std::size_t dimension>
basis_function_type<CoordinatePrecision, CoefficientPrecision, dimension> make_p0_basis_on_triangle() {
    return [](
               std::vector<std::array<CoordinatePrecision, dimension>> /*vertices*/,
               int i,
               std::array<CoordinatePrecision, dimension> /*x*/)
               -> CoefficientPrecision {
        if (i != 0) {
            throw std::out_of_range(
                "P0 basis index must be 0");
        }

        return CoefficientPrecision(1);
    };
}

template <typename CoordinatePrecision, typename CoefficientPrecision, int dimension>
basis_function_type<CoordinatePrecision, CoefficientPrecision, dimension> make_p1_basis_on_triangle() {
    return [](
               std::vector<std::array<CoordinatePrecision, dimension>> vertices,
               int i,
               std::array<CoordinatePrecision, dimension> x)
               -> CoefficientPrecision {
        static_assert(
            dimension == 2 || dimension == 3,
            "Triangle must live in 2D or 3D");

        const auto &A = vertices[0];
        const auto &B = vertices[1];
        const auto &C = vertices[2];

        auto dot = [](const auto &u, const auto &v) {
            CoordinatePrecision result = 0;

            for (std::size_t d = 0; d < dimension; ++d)
                result += u[d] * v[d];

            return result;
        };

        auto sub = [](const auto &u, const auto &v) {
            std::array<CoordinatePrecision, dimension> out{};

            for (std::size_t d = 0; d < dimension; ++d)
                out[d] = u[d] - v[d];

            return out;
        };

        const auto e1 = sub(B, A);
        const auto e2 = sub(C, A);
        const auto r  = sub(x, A);

        // Solve:
        // x = A + xi e1 + eta e2

        const auto g11 = dot(e1, e1);
        const auto g12 = dot(e1, e2);
        const auto g22 = dot(e2, e2);

        const auto b1 = dot(r, e1);
        const auto b2 = dot(r, e2);

        const auto det = g11 * g22 - g12 * g12;

        const auto xi =
            (g22 * b1 - g12 * b2) / det;

        const auto eta =
            (g11 * b2 - g12 * b1) / det;

        const auto lambda0 =
            CoefficientPrecision(1) - xi - eta;

        const auto lambda1 =
            CoefficientPrecision(xi);

        const auto lambda2 =
            CoefficientPrecision(eta);

        switch (i) {
        case 0:
            return lambda0;
        case 1:
            return lambda1;
        case 2:
            return lambda2;
        default:
            throw std::out_of_range(
                "P1 basis index must be 0,1,2");
        }
    };
}
