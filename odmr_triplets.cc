#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Geometry>
#include <vector>

using namespace Eigen;
using namespace std;
typedef complex<double> complexg; 
const complexg iii(0,1);

inline double sq(double x) { return x*x; }
inline double min(double x, double y) { return (y < x) ? y : x; }
inline double scalar(complexg a, complexg b) { return real(a)*real(b)+imag(a)*imag(b); }
double myrand(void) { return (double) rand() / (double) RAND_MAX; }
typedef long long int lhint;

Vector3d random_unit_vector(void) { 
   double phi = 2.0 * M_PI * myrand();
   double z = 2.0 * myrand() - 1.0;
   double r = sqrt(1.0 - z*z);

   Vector3d u; 
   u[2] = z;
   u[1] = r * sin(phi);
   u[0] = r * cos(phi);
   return u;
}

//typedef Matrix< complexg, 3, 3> Matrix3cf; 
//typedef Matrix< double, 3, 1> Vector3d; 

typedef Matrix<complexg, 9, 9> Matrix9cd; 
typedef Matrix<double, 9, 1> Vector9d; 




class Rotation {

    Matrix3d euler_matrix_z1x2z3(double z1, double x2, double z3) { 
       double cosz1 = cos(z1);
       double sinz1 = sin(z1);
       Matrix3d Z1;
       Z1 << cosz1, -sinz1, 0.0,
	 sinz1, cosz1, 0.0,
	 0.0, 0.0, 1.0;

       double cosx2 = cos(x2);
       double sinx2 = sin(x2);
       Matrix3d X2;
       X2 << 1.0, 0.0, 0.0,
	 0.0, cosx2, -sinx2,
	 0.0, sinx2, cosx2;


       double cosz3 = cos(z3);
       double sinz3 = sin(z3);
       Matrix3d Z3;
       Z3 << cosz3, -sinz3, 0.0,
	 sinz3, cosz3, 0.0,
	 0.0, 0.0, 1.0;

       return Z1 * X2 * Z3;
    }

    void mat2euler(const Matrix3d &M, Vector3d &vec) {
       std::numeric_limits<double> double_limit;
       double sy_thresh = double_limit.min() * 4.;
       double sy = sqrt(M(2,0)*M(2,0) + M(2,1)*M(2,1));
       double z1, x2, z3;
       if (sy > sy_thresh) { 
	  x2 = acos(M(2,2));
	  z1 = atan2(M(0,2), -M(1,2));
	  z3 = atan2(M(2,0), M(2,1));
       } else {
	  x2 = 0;
	  z3 = 0;
	  z1 = atan2(M(1,0), M(1,1));
       }
       vec << z1, x2, z3;
    }

    Vector3d angles;
    Matrix3d M;

    void init_from_angles(double z1, double x2, double z3) { 
       angles << z1, x2, z3;
       M = euler_matrix_z1x2z3(z1, x2, z3);
    }

    void init_from_matrix(const Matrix3d &Mnew) {
       M = Mnew;
       mat2euler(M, angles);
    }

public : 

    const Matrix3d &matrix(void) {
       return M;
    }

    const Matrix3d &matrix(const Matrix3d &Mnew) {
       init_from_matrix(Mnew);
       return M;
    }

    Rotation(void) { 
       init_from_angles(0, 0, 0);
    }

    Rotation(double z1, double x2, double z3) { 
       init_from_angles(z1, x2, z3);
    }

    Rotation(const Vector3d &v) { 
       init_from_angles(v[0], v[1], v[2]);
    }

    Rotation(const Matrix3d &Mnew) { 
       init_from_matrix(Mnew);
    }

    void operator = (const Matrix3d &Mnew) { 
       init_from_matrix(Mnew);
    }

    const Vector3d &euler_angles(void) {
       return angles;
    }

    const Vector3d &euler_angles(double z1, double x2, double z3) {
       init_from_angles(z1, x2, z3);
       return angles;
    }

    const Vector3d &euler_angles(const Vector3d &v) {
       init_from_angles(v[0], v[1], v[2]);
       return angles;
    }        


    // allows assignements like 
    // Rotation R =  (Rotation::Y(phi) * Rotation::X(theta)).eval();
    static Matrix3d X(double angle) { 
       Vector3d ux;
       ux << 1.0, 0.0, 0.0;
       Matrix3d R;
       R = AngleAxis<double> ( angle, ux);
       return R;
    }

    static Matrix3d Y(double angle) { 
       Vector3d uy;
       uy << 0.0,1.0,0.0;
       Matrix3d R;
       R = AngleAxis<double> ( angle, uy);
       return R;
    }

    static Matrix3d Z(double angle) { 
       Vector3d uz;
       uz << 0.0,0.0,1.0;
       Matrix3d R;
       R = AngleAxis<double> ( angle, uz);
       return R;
    }


    void random(void) { 
       M = AngleAxis<double> ( 2.0 * M_PI * myrand(), random_unit_vector() );
       init_from_matrix(M);
    }
};






class TripletHamiltonian { 
public:
    Matrix3cd Sx;
    Matrix3cd Sy;
    Matrix3cd Sz;
    Matrix3cd Id;

    TripletHamiltonian(void) { 
       Id << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
       Sz << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0;
       Sx << 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0;
       Sx /= sqrt(2.0);
       Sy << 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0;
       Sy *= -iii / sqrt(2.0);
    }

    Matrix3cd zeeman(double Bx, double By, double Bz) { 
       return Bx * Sx + By * Sy + Bz * Sz;
    }

    Matrix3cd fine_structure(double D, double E,  Rotation rot) { 
       Matrix3d r_matrix = rot.matrix();
       Matrix3cd rSx = r_matrix(0, 0) * Sx + r_matrix(0, 1) * Sy + r_matrix(0, 2) * Sz;
       Matrix3cd rSy = r_matrix(1, 0) * Sx + r_matrix(1, 1) * Sy + r_matrix(1, 2) * Sz;
       Matrix3cd rSz = r_matrix(2, 0) * Sx + r_matrix(2, 1) * Sy + r_matrix(2, 2) * Sz;
       return D * (rSz * rSz - 2.0*Id/3.0) + E * (rSy * rSy -  rSx * rSx);
    }

    Matrix3cd fine_structure(double D, double E) { 
       return D * (Sz * Sz - 2.0*Id/3.0) + E * (Sy * Sy -  Sx * Sx);
    }

    Matrix3cd spin_hamiltonian_mol_basis(double D, double E, double B, double theta, double phi) { 
       double Bx = B * sin(theta) * cos(phi);
       double By = B * sin(theta) * sin(phi);
       double Bz = B * cos(theta);
       return fine_structure(D, E) + Sx * Bx + Sy * By + Sz * Bz;
    }

    Matrix3cd spin_hamiltonian_field_basis(double D, double E, double B, double theta, double phi) { 
       return fine_structure(D, E, Rotation(0, -theta, -phi + M_PI/2.0)) + Sz * B;
    }
};



class TwoTriplets { 
    TripletHamiltonian triplet;
    Matrix9cd Hfull; // Hamiltonian 
    Matrix9cd Jproj;
    Matrix9cd tensor_product(const Matrix3cd &A, const Matrix3cd &B) { 
       return kroneckerProduct(A, B).eval();
    }

    Matrix9cd exchange_matrix(void) { 
       return kroneckerProduct(triplet.Sx, triplet.Sx).eval() + kroneckerProduct(triplet.Sy, triplet.Sy).eval() + kroneckerProduct(triplet.Sz, triplet.Sz).eval();
    }

    Matrix9cd dipole_dipole_matrix(Vector3d uvec) {       
       double unorm = uvec.norm();
       if (abs(unorm) > 1e-15) { 
	  uvec /= unorm;
	  Matrix3cd uS = uvec(0) * triplet.Sx + uvec(1) * triplet.Sy + uvec(2) * triplet.Sz;       
	  return (exchange_matrix() - 3.0 * tensor_product(uS, uS));
       } else {
	  return Matrix9cd::Zero();
       }
    }

public : 
    enum { matrix_size = 9 };
    double D;
    double E;
    double J;
    double Jdip;
    double B;
    Vector9d eval;   // eigenvalues
    Matrix9cd evec;  // eigenvectors

    TwoTriplets(void) : triplet() { 

       //       Matrix9cd J2;
       //       load_exchange_matrix(J2);
       //       SealfAdjointEigenSolver<Matrix9cd> eigensolver(J2);
       //       if (eigensolver.info() != Success) abort();
       //       Jproj = eigensolver.eigenvectors().transpose();

       double s2i3 = sqrt(2.0/3.0);
       double si2 = 1.0/sqrt(2.0);
       double si3 = 1.0/sqrt(3.0);
       double si6 = 1.0/sqrt(6.0);
       Jproj << 0, 0, si3, 0, -si3, 0, si3, 0, 0,
	 0, 0, 0, 0, 0, -si2, 0, si2, 0,
	 0, 0, -si2, 0, 0, 0, si2, 0, 0,
	 0, -si2, 0, si2, 0, 0, 0, 0, 0,
	 0,    0, 0,   0, 0, 0, 0, 0, 1.0, 
	 0, 0, 0, 0, 0, si2, 0, si2, 0,
	 0, 0, si6, 0, s2i3, 0, si6, 0, 0,
	 0, si2, 0, si2, 0, 0, 0, 0, 0,
	 1.0, 0, 0, 0, 0, 0, 0, 0, 0;


    }


    void load_field_basis_Hamiltonian( Rotation &triplet1_rotation,  Rotation &triplet2_rotation) {
       Matrix3cd H1 = triplet.fine_structure(D, E, triplet1_rotation) + B * triplet.Sz; 

       Matrix3cd H2 = triplet.fine_structure(D, E, triplet2_rotation) + B * triplet.Sz; 

       Hfull = tensor_product(H1, triplet.Id) + tensor_product(triplet.Id, H2) + J * exchange_matrix();
    }

    void load_field_basis_Hamiltonian( Rotation &triplet1_rotation,  Rotation &triplet2_rotation, Vector3d rdip) {
       load_field_basis_Hamiltonian(triplet1_rotation, triplet2_rotation);
       Hfull += Jdip * dipole_dipole_matrix(rdip);
    }
    

    void load_mol1_basis_Hamiltonian(double theta, double phi,  Rotation &triplet2_rotation) {
       double Bx = B * sin(theta) * cos(phi);
       double By = B * sin(theta) * sin(phi);
       double Bz = B * cos(theta);

       Matrix3cd Hz = triplet.zeeman(Bx, By, Bz);
       Matrix3cd H1 = triplet.fine_structure(D, E) + Hz;
       Matrix3cd H2 = triplet.fine_structure(D, E, triplet2_rotation) + Hz; 

       Hfull = tensor_product(H1, triplet.Id) + tensor_product(triplet.Id, H2) + J * exchange_matrix();
    }
    

    void diag(void) { 
       SelfAdjointEigenSolver<Matrix9cd> eigensolver(Hfull);
       if (eigensolver.info() != Success) abort();
       eval = eigensolver.eigenvalues();
       evec = eigensolver.eigenvectors();
    }

 
    double sz_elem(int i) { 
       Matrix9cd Sz2 = kroneckerProduct(triplet.Sz, triplet.Id).eval() + kroneckerProduct(triplet.Id, triplet.Sz).eval();
       Matrix<complexg, 9, 1> vi = evec.col(i);
       Matrix<complexg, 1, 1> Sz2ii = vi.adjoint() * Sz2 * vi;
       return real(Sz2ii(0));
    }

    double quintet_content(int i) {
       Matrix<complexg, 5, 1> iProj = Jproj.block(4, 0, 5, 9) * evec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    double triplet_content(int i) {
       Matrix<complexg, 3, 1> iProj = Jproj.block(1, 0, 3, 9) * evec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    double singlet_content(int i) {
       Matrix<complexg, 1, 1> iProj = Jproj.block(0, 0, 1, 9) * evec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    Matrix9cd singlet_projector(void) { 
       return Jproj.row(0).adjoint() * Jproj.row(0);
    }

    Matrix9cd Bac_field_basis_matrix(void) { 
       return tensor_product(triplet.Sx, triplet.Id) + tensor_product(triplet.Id, triplet.Sx);
    }

    Matrix9cd hamiltonian(void) const { 
       return Hfull;
    }

    void print_info(void) { 
      cout << "# D " << D << endl;
      cout << "# E " << E << endl;
      cout << "# B " << B << endl;
      cout << "# J " << J << endl;
      cout << "# Jdip " << Jdip << endl;
    }

};


/*
 * ODMR_Signal
 *
 * Output : Computes ODMR and magnetic resonance signals 
 *
 * Input : spins, a reference on SpinSystem object
 * SpinSystem should define 
 * spins.matrix_size
 * spins.evec
 * spins.eval
 * spins.singlet_projector()
 * spins.Bac_field_basis_matrix()
 */ 

template <class SpinSystem> class ODMR_Signal { 
    typedef Matrix<double, SpinSystem::matrix_size, 1> SpinVectord;
    typedef Matrix<complexg, SpinSystem::matrix_size, SpinSystem::matrix_size> SpinMatrixcd;

    SpinVectord rho0;
    SpinMatrixcd rho2;
    SpinMatrixcd Sproj_eig_basis;
    SpinMatrixcd V;
    SpinSystem &spins;

public : 
    double gamma;
    double gamma_diag;


    ODMR_Signal(SpinSystem &spin_system) : spins(spin_system) {

    }

    void update_from_spin_hamiltonian(void) { 
        Sproj_eig_basis = spins.evec.adjoint() * spins.singlet_projector() * spins.evec;
	V = spins.evec.adjoint() * spins.Bac_field_basis_matrix() * spins.evec; 
    }

    double omega_nm(int n, int m) { 
       return spins.eval(n) - spins.eval(m);
    }

    void load_rho0_thermal(double Temp) { 
       for (int i = 0; i < spins.matrix_size ; i++) { 
	  rho0(i) = exp(- spins.eval(i) / Temp);
       }
       double t = rho0.sum();
       rho0 /= t;
    }

    void load_rho0_from_singlet(void) { 
       for (int i = 0; i < spins.matrix_size ; i++) { 
	  rho0(i) = real(Sproj_eig_basis(i, i));
       }
       double t = rho0.sum();
       rho0 /= t;
    }    

    complexg chi1(double omega) { 
       complexg c1 = 0.0;
       for (int m = 0; m < spins.matrix_size ; m++) { 
	  for (int n = 0; n < spins.matrix_size ; n++) { 
	     // 
	     // the contribution to chi1 vanishes for n == m, whether gamma is the same for diagonal and non diagonal elements is not relvant here 
	     // 
	     // cerr << n << "    " << m << "    " << abs(V(m, n)) << endl;
	     c1 -= (rho0(m) - rho0(n)) * norm(V(n, m)) / ( omega_nm(n, m) - omega - iii * gamma );
	  }
       }
       return c1;
    }    


    // explicit calculation of rho2 - close to analytical formula but slow
    void find_rho2_explicit(double omega) { 
       for (int m = 0; m < spins.matrix_size ; m++) { 
	  for (int n = 0; n < spins.matrix_size ; n++) { 
	     complexg rrr = 0.0;
	     for (int nu = 0; nu < spins.matrix_size ; nu++) { 
	        for (int p = -1; p <= 1; p += 2) { 
		  // Vtmp(nu, m) = (rho0(m) - rho0(nu)) * V(nu, m) / ( omega_nm(nu, m) - omega * (double) p - iii * gamma )
		   rrr += V(n, nu) * (rho0(m) - rho0(nu)) * V(nu, m) / ( omega_nm(nu, m) - omega * (double) p - iii * gamma );
		  //  nu->n and m->nu : Vtmp(n, nu)  
		   rrr -= ((rho0(nu) - rho0(n)) * V(n, nu) / ( omega_nm(n, nu) - omega * (double) p - iii * gamma )) * V(nu, m);
		}
	     }
	     // relaxation may be different for diaganonal and non diagonal terms
	     double gamma_nm = (n == m) ? gamma_diag : gamma;
	     rho2(n, m) = rrr / ( omega_nm(n, m) - iii * gamma_nm );
	  }
       }

    }


    // optimized calculation of rho2
    void find_rho2(double omega) { 
       SpinMatrixcd Vtmp = SpinMatrixcd::Zero();
       for (int m = 0; m < spins.matrix_size ; m++) { 
	  for (int nu = 0; nu < spins.matrix_size ; nu++) { 
	     for (int p = -1; p <= 1; p += 2) { 
	        Vtmp(nu, m) += (rho0(m) - rho0(nu)) * V(nu, m) / (omega_nm(nu, m) - omega * (double) p - iii * gamma);
	     }
	  }
       }      
       rho2 = V * Vtmp - Vtmp * V;
       for (int m = 0; m < spins.matrix_size ; m++) { 
	  for (int n = 0; n < spins.matrix_size ; n++) { 
	     // relaxation may be different for diaganonal and non diagonal terms
	     double gamma_nm = (n == m) ? gamma_diag : gamma;
	     rho2(n, m) /= ( omega_nm(n, m) - iii * gamma_nm );
	  }
       }
    }


    double odmr(double omega) { 
       double odmr_amp = 0.0;
       find_rho2(omega);
       
       for (int m = 0; m < spins.matrix_size ; m++) { 
	  for (int n = 0; n < spins.matrix_size ; n++) { 
	     odmr_amp += real( rho2(m , n) * Sproj_eig_basis(n, m) );
	  }
       }

       return odmr_amp;
    }
};


/*
 * Merrifield
 *
 * Computes steady state spin density matrix from the master equation 
 * using eigen's matrix free iterative solvers.
 * Matrix free solver code borrowed from  
 * https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html
 *
 * Input : spins, a reference on SpinSystem object
 * SpinSystem should define 
 * spins.matrix_size
 * spins.singlet_projector()
 * spins.hamiltonian()
 *
 * function PL_from_rate() uses spins.singlet_content 
 *
 */ 
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

template<typename SpinSystem> class Merrifield;

namespace Eigen {
namespace internal {
  // make Merrifield look like Matrix<complexg, SpinSystem::matrix_size^2, SpinSystem::matrix_size^2> 
  template<class SpinSystem>
  //  struct traits< Merrifield<SpinSystem> > : public Eigen::internal::traits< Matrix<complexg, SpinSystem::matrix_size*SpinSystem::matrix_size, SpinSystem::matrix_size*SpinSystem::matrix_size> >
  struct traits< Merrifield<SpinSystem> > : public Eigen::internal::traits<Eigen::SparseMatrix<complexg> >
  {};

}
}

template <class SpinSystem> class Merrifield :public Eigen::EigenBase< Merrifield<SpinSystem> >  { 
    SpinSystem &spins;
public:
    double gammaS;
    double gamma;
    double gen;
    

    // Required typedefs, constants, and method:
    typedef complexg Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    typedef Matrix<complexg, SpinSystem::matrix_size, SpinSystem::matrix_size> SpinMatrix;
    typedef Matrix<complexg, SpinSystem::matrix_size*SpinSystem::matrix_size, 1> SpinMatrixVecForm;
    SpinMatrix Ps;
    SpinMatrix rho;

    enum {
       ColsAtCompileTime = Eigen::Dynamic,
       MaxColsAtCompileTime = Eigen::Dynamic,
       IsRowMajor = false
    };
  
    Index rows() const { return SpinSystem::matrix_size * SpinSystem::matrix_size; }
    Index cols() const { return SpinSystem::matrix_size * SpinSystem::matrix_size; }

    template<typename Rhs>
    Eigen::Product<Merrifield<SpinSystem>,Rhs,Eigen::AliasFreeProduct> 
    operator*(const Eigen::MatrixBase<Rhs>& x) const {
       return Eigen::Product<Merrifield<SpinSystem>,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
    }


    Merrifield(SpinSystem &spin_system) : spins(spin_system) {
       Ps = spins.singlet_projector();
    }

    SpinMatrix Liouvillian(const SpinMatrix &rho) const { 
      SpinMatrix L = -iii * ( spins.hamiltonian() * rho - rho * spins.hamiltonian() )
	- gamma * rho
	- gammaS * (Ps * rho + rho * Ps);
      return L;
    }


    SpinMatrix map_to_mat(SpinMatrixVecForm vec) const { 
        return Map< SpinMatrix >(vec.data());
    }

    SpinMatrixVecForm map_to_vec(SpinMatrix &mat) const { 
        return Map< SpinMatrixVecForm > (mat.data());
    }

    SpinMatrixVecForm Ps_to_vec(void) { 
        return map_to_vec(Ps);
    }

    void find_rho(void) { 
        Eigen::BiCGSTAB< Merrifield<SpinSystem> , Eigen::IdentityPreconditioner> bicg;
	bicg.compute(*this);
	SpinMatrixVecForm x;
	SpinMatrixVecForm y = -Ps_to_vec();
	x = bicg.solve(y);    
	//	std::cout << "BiCGSTAB: #iterations: " << bicg.iterations() << ", estimated error: " << bicg.error() << std::endl;
	rho = map_to_mat(x);
    }


    double rho_error(void) { 
        return (Liouvillian(rho) + Ps).norm();
    }

    double PL(void) { 
	Matrix<complexg, 1, 1> sum;
	for (int i = 0; i < SpinSystem::matrix_size; i++) {
	   sum += Ps.row(i) * rho.col(i);
	}
	return real(sum(0));
    }


    double PL_from_rate(void) { 
        double sum = 0.0;
	for (int i = 0; i < SpinSystem::matrix_size; i++) {
	   double alpha_n = spins.singlet_content(i);
	   sum += alpha_n * alpha_n / (gamma + gammaS * alpha_n);
	}
	return sum;
    }

};


namespace Eigen {
namespace internal {
  template<typename Rhs, class SpinSystem>
  struct generic_product_impl<Merrifield<SpinSystem>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<Merrifield<SpinSystem>,Rhs,generic_product_impl<Merrifield<SpinSystem>,Rhs> >
  {
    typedef typename Product<Merrifield<SpinSystem>,Rhs>::Scalar Scalar;
    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const Merrifield<SpinSystem>& lhs, const Rhs& rhs, const Scalar& alpha)
    {
      // This method should implement "dst += alpha * lhs * rhs" inplace,
      // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
      assert(alpha==Scalar(1) && "scaling is not implemented");
      typename Merrifield<SpinSystem>::SpinMatrix rho = lhs.map_to_mat(rhs);     
      typename Merrifield<SpinSystem>::SpinMatrix L = lhs.Liouvillian(rho);
      typename Merrifield<SpinSystem>::SpinMatrixVecForm lhs_x_rhs = lhs.map_to_vec(L);
      dst += lhs_x_rhs;
    }
  };
}
}

//
// demonstration code for Merrifield class with both Liouville and rate equation versions
//
int main()
{
    TwoTriplets triplet_pair;
    triplet_pair.D = 0.1;
    triplet_pair.E = 0.0;
    triplet_pair.J = 5.0;
    triplet_pair.Jdip = 0.0;
    triplet_pair.B = 5.0;
    triplet_pair.print_info();
    
    Rotation triplet1_rot, triplet2_rot;
    triplet1_rot.random();
    triplet2_rot.random();
    Vector3d rdip = random_unit_vector();


    Merrifield<TwoTriplets> merrifield(triplet_pair);
    merrifield.gammaS = 0.1;
    merrifield.gamma = 0.15;
    
    for (double B = 0.0; B < 30; B += 0.01) { 
       triplet_pair.B = B;
       triplet_pair.load_field_basis_Hamiltonian(triplet1_rot, triplet2_rot, rdip);
       triplet_pair.diag(); // needed for PL_from_rate()
       merrifield.find_rho();
       double PL = merrifield.PL();
       cout << triplet_pair.B << "     " << PL << "    " << merrifield.PL_from_rate() << "     " << merrifield.rho_error() << endl;
    }

}

//
// demonstration code for MR/ODMR colormap as function of B and frequency 
//
int main_odmr()
{
    TwoTriplets triplet_pair;
    triplet_pair.D = 1.0;
    triplet_pair.E = 0.15;
    triplet_pair.J = 0.0;
    triplet_pair.Jdip = 0.03;
    triplet_pair.B = 5.0;
    triplet_pair.print_info();

    srand(1);

    double quintet_max = 0.0;

    Rotation quintet_t1_rot;
    Rotation quintet_t2_rot;
    Vector3d quintet_rdip = random_unit_vector();

    int Nsamples = 5000;
    std::vector<double> slist(Nsamples);

    double theta = 1.1;
    for (int count = 0; count < Nsamples; count++) { 
       Rotation triplet1_rot, triplet2_rot;
       triplet1_rot = (Rotation::Y(0) * Rotation::X(theta)).eval();
       triplet2_rot.random();
       Vector3d rdip  = random_unit_vector();
       triplet_pair.load_field_basis_Hamiltonian(triplet1_rot, triplet2_rot, rdip);
       triplet_pair.diag(); 
       double si = 0.0;
       for (int i = 0; i < triplet_pair.matrix_size ; i++) { 
	  double qi = triplet_pair.quintet_content(i);
	  si += pow(qi, 4.0);
       }
       if (si > quintet_max) { 
	  quintet_max = si;
	  quintet_t1_rot = triplet1_rot;
	  quintet_t2_rot = triplet2_rot;
	  quintet_rdip = rdip;
       }
       slist[count] = si;
    }

    cout << "# quintet_max " << quintet_max << endl;
    cout << "# triplet Euler angles :" << endl;
    cout << "# ";
    for (int i = 0; i < 3; i++) cout << quintet_t1_rot.euler_angles()[i] << "   ";
    cout << endl;
    cout << "# ";
    for (int i = 0; i < 3; i++) cout << quintet_t2_rot.euler_angles()[i] << "   ";
    cout << endl;
    cout << "# Rdip :" << endl;
    cout << "# ";
    for (int i = 0; i < 3; i++) cout << quintet_rdip[i] << "   ";
    cout << endl;

    triplet_pair.load_field_basis_Hamiltonian(quintet_t1_rot, quintet_t2_rot, quintet_rdip);
    triplet_pair.diag();
    cout << "# qunitet/triplet projections at B = " << triplet_pair.B << endl;
    for (int i = 0; i < triplet_pair.matrix_size ; i++) 
      cout << "# " << triplet_pair.quintet_content(i) << "    " << triplet_pair.triplet_content(i) << "    " << triplet_pair.singlet_content(i) << "    " << triplet_pair.sz_elem(i) << endl;


    double cos1z = quintet_t1_rot.matrix()(2,2);
    double cos2z = quintet_t2_rot.matrix()(2,2);
    cout << "# angles to field " << endl;
    cout << "# " << theta << "   " << "    " << quintet_max << "   "  << cos1z << "    " << cos2z << "    " << endl;

    ODMR_Signal<TwoTriplets> odmr_from_triplets(triplet_pair);    

    const int N_averages = 10000;
    double omega_span = 10.0;     
    const int n_omega_samples = 1000;

    double B = 5.0;
    //    for (double B = 0; B < 3.0; B += 0.01) { 

       vector<complexg> chi_B(n_omega_samples, 0.0);
       vector<double> odmr_B(n_omega_samples, 0.0);
    
       for (int sample = 0; sample < N_averages; sample++) {        
	  Rotation rot1;
	  rot1.random();
	  Rotation r1_sample = (rot1.matrix() * quintet_t1_rot.matrix()).eval();
	  Rotation r2_sample = (rot1.matrix() * quintet_t2_rot.matrix()).eval();
	  Vector3d rdip_sample = rot1.matrix() * random_unit_vector(); // rot1.matrix() * quintet_rdip;
	  triplet_pair.B = B;
	  triplet_pair.load_field_basis_Hamiltonian(r1_sample, r2_sample, rdip_sample);
	  triplet_pair.diag();

	  odmr_from_triplets.update_from_spin_hamiltonian();
	  //	  odmr_from_triplets.load_rho0_thermal(10.0);
	  odmr_from_triplets.load_rho0_from_singlet();
	  odmr_from_triplets.gamma = 3e-3;
	  odmr_from_triplets.gamma_diag = 3e-3;

	  for (int omega_index = 0; omega_index < n_omega_samples; omega_index++) { 
	    double omega = omega_span * (double)omega_index/(double)(n_omega_samples-1);
	    chi_B[omega_index] += odmr_from_triplets.chi1(omega)  / (double) N_averages;
	    odmr_B[omega_index] += odmr_from_triplets.odmr(omega) / (double) N_averages;
	  }
       }


       for (int omega_index = 0; omega_index < n_omega_samples; omega_index++) { 
	 double omega = omega_span * (double)omega_index/(double)(n_omega_samples-1);
	 cout  << B << "    " << omega << "   " << real(chi_B[omega_index]) << "    " << imag(chi_B[omega_index]) << "     " << odmr_B[omega_index] << endl;
       }

       cout << endl;
       return 0;
}
