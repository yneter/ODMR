#include <iostream>
#include <Eigen/Dense>
#include <Eigen/KroneckerProduct>
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

    Matrix3cd zeeman(double Bx, double By, double Bz) { 
       return Bx * Sx + By * Sy + Bz * Sz;
    }

    Matrix3cd fine_structure(double D, double E, double z1, double x2, double z3) { 
       Matrix3d r_matrix = euler_matrix_z1x2z3(z1, x2, z3);
       Matrix3cd rSx = r_matrix(0, 0) * Sx + r_matrix(0, 1) * Sy + r_matrix(0, 2) * Sz;
       Matrix3cd rSy = r_matrix(1, 0) * Sx + r_matrix(1, 1) * Sy + r_matrix(1, 2) * Sz;
       Matrix3cd rSz = r_matrix(2, 0) * Sx + r_matrix(2, 1) * Sy + r_matrix(2, 2) * Sz;
       return D * (rSz * rSz - 2.0*Id/3.0) + E * (rSy * rSy -  rSx * rSx);
    }


    Matrix3cd spin_hamiltonian_mol_basis(double D, double E, double B, double theta, double phi) { 
       double Bx = B * sin(theta) * cos(phi);
       double By = B * sin(theta) * sin(phi);
       double Bz = B * cos(theta);
       return fine_structure(D, E, 0, 0, 0) + Sx * Bx + Sy * By + Sz * Bz;
    }

    Matrix3cd spin_hamiltonian_field_basis(double D, double E, double B, double theta, double phi) { 
       return fine_structure(D, E, 0, -theta, -phi + M_PI/2.0) + Sz * B;
    }
};



class TwoTriplets { 
    TripletHamiltonian triplet;

    Matrix9cd Hfull;
    Vector9d Heval;
    Matrix9cd Hevec;
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
    double D;
    double E;
    double J;
    double Jdip;
    double B;


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


    void load_field_basis_Hamiltonian(Vector3d triplet1_angles, Vector3d triplet2_angles) {
       Matrix3cd H1 = triplet.fine_structure(D, E, triplet1_angles[0], triplet1_angles[1], triplet1_angles[2]) + B * triplet.Sz; 

       Matrix3cd H2 = triplet.fine_structure(D, E, triplet2_angles[0], triplet2_angles[1], triplet2_angles[2]) + B * triplet.Sz; 

       Hfull = tensor_product(H1, triplet.Id) + tensor_product(triplet.Id, H2) + J * exchange_matrix();
    }

    void load_field_basis_Hamiltonian(Vector3d triplet1_angles, Vector3d triplet2_angles, Vector3d rdip) {
       load_field_basis_Hamiltonian(triplet1_angles, triplet2_angles);
       Hfull += Jdip * dipole_dipole_matrix(rdip);
    }
    

    void load_mol1_basis_Hamiltonian(double theta, double phi, Vector3d triplet2_angles) {
       double Bx = B * sin(theta) * cos(phi);
       double By = B * sin(theta) * sin(phi);
       double Bz = B * cos(theta);

       Matrix3cd Hz = triplet.zeeman(Bx, By, Bz);
       Matrix3cd H1 = triplet.fine_structure(D, E, 0, 0, 0) + Hz;
       Matrix3cd H2 = triplet.fine_structure(D, E, triplet2_angles[0], triplet2_angles[1], triplet2_angles[2]) + Hz; 

       Hfull = tensor_product(H1, triplet.Id) + tensor_product(triplet.Id, H2) + J * exchange_matrix();
    }
    

    void diag(void) { 
       SelfAdjointEigenSolver<Matrix9cd> eigensolver(Hfull);
       if (eigensolver.info() != Success) abort();
       Heval = eigensolver.eigenvalues();
       Hevec = eigensolver.eigenvectors();
    }

    double eval(int i) { 
       return Heval(i);
    };

  
    double sz_elem(int i) { 
       Matrix9cd Sz2 = kroneckerProduct(triplet.Sz, triplet.Id).eval() + kroneckerProduct(triplet.Id, triplet.Sz).eval();
       Matrix<complexg, 9, 1> vi = Hevec.col(i);
       Matrix<complexg, 1, 1> Sz2ii = vi.adjoint() * Sz2 * vi;
       return real(Sz2ii(0));
    }

    double quintet_content(int i) {
       Matrix<complexg, 5, 1> iProj = Jproj.block(4, 0, 5, 9) * Hevec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    double triplet_content(int i) {
       Matrix<complexg, 3, 1> iProj = Jproj.block(1, 0, 3, 9) * Hevec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    double singlet_content(int i) {
       Matrix<complexg, 1, 1> iProj = Jproj.block(0, 0, 1, 9) * Hevec.block(0, i, 9, 1);
       Matrix<complexg, 1, 1> norm2 = iProj.adjoint() * iProj;
       return real(norm2(0));
    }

    int size(void) { 
       return Heval.size();
    }
   
    void print_info(void) { 
      cout << "# D " << D << endl;
      cout << "# E " << E << endl;
      cout << "# B " << B << endl;
      cout << "# J " << J << endl;
      cout << "# Jdip " << Jdip << endl;
    }

};



int main()
{
  //    NTriplets Exciton(2);
    TwoTriplets Exciton;
    Exciton.D = 1.0;
    Exciton.E = 0.3;
    Exciton.J = 0.0;
    Exciton.Jdip = 0.1;
    Exciton.B = 5.0;
    Exciton.print_info();

    srand(1);

    double quintet_max = 0.0;

    Vector3d quintet_angles1;
    Vector3d quintet_angles2;
    Vector3d quintet_rdip;

    int Nsamples = 5000;
    std::vector<double> slist(Nsamples);
    for (int count = 0; count < Nsamples; count++) { 
       Vector3d angles1;
       Vector3d angles2;
       Vector3d rdip;
       for (int i=0; i<3; i++) { 
	  angles1[i] = 2.0 * M_PI * myrand();
	  angles2[i] = 2.0 * M_PI * myrand();
       }
       rdip = random_unit_vector();
       Exciton.load_field_basis_Hamiltonian(angles1, angles2, rdip );
       Exciton.diag(); 

       double si = 0.0;
       for (int i = 0; i < Exciton.size() ; i++) { 
	  double qi = Exciton.quintet_content(i);
	  si += pow(qi, 4.0);
	 //	 double s2i = Exciton.sz_elem(i);
	 //	 si += pow(s2i, 4.0);
       }
       if (si > quintet_max) { 
	  quintet_max = si;
	  quintet_angles1 = angles1;
	  quintet_angles2 = angles2;
	  quintet_rdip = rdip;
       }
       slist[count] = si;
    }


    cout << "# quintet_max " << quintet_max << endl;
    // quintet_angles1 << 5.8308, 4.34636, 3.31015;
    // quintet_angles2 << 2.88627, 4.83054, 2.48384;
    // quintet_rdip << 0.0645974, 0.0136964, 0.997817;

    cout << "# triplet Euler angles :" << endl;
    for (int i = 0; i < 3; i++) cout << quintet_angles1[i] << "   ";
    cout << endl;
    for (int i = 0; i < 3; i++) cout << quintet_angles2[i] << "   ";
    cout << endl;
    cout << "# Rdip :" << endl;
    for (int i = 0; i < 3; i++) cout << quintet_rdip[i] << "   ";
    cout << endl;
    
    Exciton.load_field_basis_Hamiltonian(quintet_angles1, quintet_angles2, quintet_rdip);
    Exciton.diag();
    cout << "# qunitet/triplet projections at B = " << Exciton.B << endl;
    for (int i = 0; i < Exciton.size() ; i++) 
      cout << Exciton.quintet_content(i) << "    " << Exciton.triplet_content(i) << "    " << Exciton.singlet_content(i) << "    " << Exciton.sz_elem(i) << endl;
}
