#include <iostream>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/cusolver.hpp>
#include <cutf/curand.hpp>

#include <matfile/matfile.hpp>

template <class T>
std::string get_dtype_name_str();
template <>
std::string get_dtype_name_str<float >() {return "sp";}
template <>
std::string get_dtype_name_str<double>() {return "dp";}

template <class T>
__global__ void V_x_rS_x_Ut_kernel(
		const std::size_t m,
		const std::size_t n,
		const std::size_t r,
		T* const ra_ptr, const std::size_t lda,
		const T* const u_ptr, const std::size_t ldu,
		const T* const s_ptr,
		const T* const vt_ptr, const std::size_t ldvt
		) {
	const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= m * n) {
		return;
	}

	const auto mi = tid % m;
	const auto ni = tid / m;

	T ra = 0;
	for (unsigned ri = 0; ri < r; ri++) {
		ra += vt_ptr[ldvt * mi + ri] * (1 / s_ptr[ri]) * u_ptr[ldu * ri + ni];
	}
	ra_ptr[mi + ni * lda] = ra;
}

template <class T>
void V_x_rS_x_Ut(
		const std::size_t m,
		const std::size_t n,
		const std::size_t r,
		T* const ra_ptr, const std::size_t lda,
		const T* const u_ptr, const std::size_t ldu,
		const T* const s_ptr,
		const T* const vt_ptr, const std::size_t ldvt
		) {
	const auto block_size = 256u;
	const auto grid_size= (m * n + block_size - 1) / block_size;

	V_x_rS_x_Ut_kernel<T>
		<<<grid_size, block_size>>>(
				m, n, r,
				ra_ptr, lda,
				u_ptr, ldu,
				s_ptr,
				vt_ptr, ldvt
				);
}

template <class T>
void generate_matrix_pair(
		const std::size_t m,
		const std::size_t n,
		const std::uint64_t seed,
		const bool check = false
		) {
	const auto matrix_id = m * 100000000lu + n * 1000lu + seed;
	std::stringstream ss;
	ss << std::hex << matrix_id;
	const std::string matrix_id_hex = ss.str();
	const std::string file_name_stem = matrix_id_hex + "-" + get_dtype_name_str<T>() + "-m" + std::to_string(m) + "-n" + std::to_string(n) + "-seed" + std::to_string(seed) + ".matrix";
	const std::string inv_file_name_stem = matrix_id_hex + "-" + get_dtype_name_str<T>() + "-inv-m" + std::to_string(n) + "-n" + std::to_string(m) + "-seed" + std::to_string(seed) + ".matrix";

	{
		auto d_mat = cutf::memory::get_device_unique_ptr<T>(m * n);
		auto h_mat = cutf::memory::get_host_unique_ptr  <T>(m * n);


		auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
		CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));

		CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), d_mat.get(), m * n));

		cutf::memory::copy(h_mat.get(), d_mat.get(), m * n);
		mtk::matfile::save_dense(
				m, n,
				h_mat.get(), m,
				file_name_stem
				);

		const std::size_t num_s = std::min(m, n);
		auto dS = cutf::memory::get_device_unique_ptr<T>(num_s);
		auto dU = cutf::memory::get_device_unique_ptr<T>(m * m);
		auto dVT = cutf::memory::get_device_unique_ptr<T>(n * n);
		auto hS = cutf::memory::get_host_unique_ptr<T>(num_s);
		auto hU = cutf::memory::get_host_unique_ptr<T>(num_s * m);
		auto hVT = cutf::memory::get_host_unique_ptr<T>(num_s * n);

		auto dInfo = cutf::memory::get_device_unique_ptr<int>(1);

		auto cusolver = cutf::cusolver::dn::get_handle_unique_ptr();

		int Lwork;
		CUTF_CHECK_ERROR(cutf::cusolver::dn::gesvd_buffer_size<T>(*cusolver.get(), m, n, &Lwork));

		auto dLwork_buffer = cutf::memory::get_device_unique_ptr<T>(Lwork);
		auto dRwork_buffer = cutf::memory::get_device_unique_ptr<T>(num_s - 1);

		CUTF_CHECK_ERROR(cutf::cusolver::dn::gesvd(
					*cusolver.get(),
					'S', 'S',
					m, n,
					d_mat.get(), m,
					dS.get(),
					dU.get(), m,
					dVT.get(), num_s,
					dLwork_buffer.get(),
					Lwork,
					dRwork_buffer.get(),
					dInfo.get()
					));

		V_x_rS_x_Ut(
				n, m, num_s,
				d_mat.get(), n,
				dU.get(), m,
				dS.get(),
				dVT.get(), num_s
				);

		cutf::memory::copy(h_mat.get(), d_mat.get(), m * n);
		mtk::matfile::save_dense(
				n, m,
				h_mat.get(), n,
				inv_file_name_stem
				);
	}
	std::printf("# Log\n");
	std::printf("A^t : %s\n", file_name_stem.c_str());
	std::printf("A   : %s\n", inv_file_name_stem.c_str());
	if (check) {
		auto mat_a_uptr = cutf::memory::get_host_unique_ptr<T>(m * n);
		auto mat_b_uptr = cutf::memory::get_host_unique_ptr<T>(m * n);
		mtk::matfile::load_dense(mat_b_uptr.get(), n, file_name_stem);
		mtk::matfile::load_dense(mat_a_uptr.get(), m, inv_file_name_stem);

		T o = 0;
#pragma omp marallel for collapse(2) reduction(+: o)
		for (unsigned i = 0; i < m; i++) {
			for (unsigned j = 0; j < m; j++) {
				T c = 0;
				for (unsigned k = 0; k < n; k++) {
					c += mat_a_uptr.get()[i + k * m] * mat_b_uptr.get()[k + j * n];
				}
				const auto diff = c - (i == j ? 1 : 0);
				o += diff * diff;
			}
		}
		o = std::sqrt(o / static_cast<T>(n));
		std::printf("orth: %e\n", o);
	}
}

int main(int argc, char** argv) {
	if (argc <= 4) {
		std::fprintf(stderr, "Usage: %s [N (N x N)] [dtype: fp32/fp64] [seed]\n", argv[0]);
		return 1;
	}

	const auto N = std::stoull(argv[1]);
	const auto dtype = std::string(argv[2]);
	const auto seed = std::stoull(argv[3]);

	if (dtype == "fp32") {
		generate_matrix_pair<float >(N, N, seed, true);
	} else if(dtype == "fp64") {
		generate_matrix_pair<double>(N, N, seed, true);
	} else {
		std::fprintf(stderr, "Error: Unknown dtype (%s)\n", dtype.c_str());
		return 1;
	}
}
