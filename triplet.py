
import numpy as np
from numpy.linalg import eig
import cmath 
import math
import sys
from functools import reduce 

def random_unit_vector() : 
    phi = 2.0 * math.pi * np.random.random()
    z = 2.0 * np.random.random() - 1.0
    r = math.sqrt(1.0 - z*z)
    return np.array([r * math.cos(phi), r * math.sin(phi), z ])



class Rotation : 
    """ 
    * Rotation : provides a representation for 3D space rotations
    * using euler angles (ZX'Z'' convention) or rotation matrices
    """
    def _euler2mat_z1x2z3(self, z1 = 0, x2 = 0, z3 = 0) :
        cosz1 = math.cos(z1)
        sinz1 = math.sin(z1)
        Z1 = np.array(
            [[cosz1, -sinz1, 0],
             [sinz1, cosz1, 0],
             [0, 0, 1]])

    
        cosx = math.cos(x2)
        sinx = math.sin(x2)
        X2 = np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]])

        cosz3 = math.cos(z3)
        sinz3 = math.sin(z3)
        Z3 = np.array(
            [[cosz3, -sinz3, 0],
             [sinz3, cosz3, 0],
             [0, 0, 1]])
            
        return reduce(np.dot, [Z1, X2, Z3] )
        

    def _mat2euler(self, M):
        M = np.asarray(M)
        try:
            sy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            sy_thresh = _FLOAT_EPS_4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        sy = math.sqrt(r31*r31 + r32*r32)
        if sy > sy_thresh: 
            x2 = math.acos(r33)
            z1 = math.atan2(r13, -r23)
            z3 = math.atan2(r31, r32)
        else:
            x2 = 0
            z3 = 0
            z1 = math.atan2(r21, r22)
        return (z1, x2, z3)

    def _init_from_angles(self, z1, x2, z3) :
        self._z1, self._x2, self._z3 = z1, x2, z3
        self._M = self._euler2mat_z1x2z3(self._z1, self._x2, self._z3)

    def _init_from_matrix(self, matrix) :
        self._M = np.asarray(matrix)
        self._z1, self._x2, self._z3 = self._mat2euler(self._M)        

    def __init__(self, arg1 = None, x2 = None, z3 = None): 
        if arg1 is None :
            self._init_from_angles(0, 0, 0) # loads identity matrix
        elif x2 is not None:
            self._init_from_angles(arg1, x2, z3)
        elif arg1.size == 3:
            self._init_from_angles(arg1[0], arg1[1], arg1[2])
        else:
            self._init_from_matrix(arg1)

    def matrix(self, new_matrix = None) : 
        if new_matrix is not None:
            self._init_from_matrix(new_matrix)
        return self._M

    def euler_angles(self, z1 = None, x2 = None, z3 = None) : 
        if z1 is not None:
            self._init_from_angles(z1, x2, z3)
        return (self._z1, self._x2, self._z3)


    def random(self) : 
        V = 2. * math.pi * np.random.rand(3)
        self.euler_angles( V[0], V[1], V[2] )



class TripletHamiltonian : 
        def __init__ (self) :
                self.Id = np.matrix('1 0 0; 0 1 0; 0 0 1', dtype=np.complex_)
                self.Sz = np.matrix('1 0 0; 0 0 0; 0 0 -1', dtype=np.complex_)
                self.Sx = np.matrix('0 1 0; 1 0 1; 0 1 0', dtype=np.complex_) / math.sqrt(2.0)
                self.Sy = - 1j * np.matrix('0 1 0; -1 0 1; 0 -1 0', dtype=np.complex_) / math.sqrt(2.0)

        def fine_structure(self, D, E, rotation = Rotation() ) :
                rotation_matrix = rotation.matrix()
                rSx = rotation_matrix[0,0] * self.Sx + rotation_matrix[0,1] * self.Sy + rotation_matrix[0,2] * self.Sz
                rSy = rotation_matrix[1,0] * self.Sx + rotation_matrix[1,1] * self.Sy + rotation_matrix[1,2] * self.Sz
                rSz = rotation_matrix[2,0] * self.Sx + rotation_matrix[2,1] * self.Sy + rotation_matrix[2,2] * self.Sz        
                return D * (np.dot(rSz, rSz) - 2.*self.Id/3.) + E * (np.dot(rSy, rSy) -  np.dot(rSx, rSx))

        def zeeman(self, Bx, By, Bz) :
                return Bx * self.Sx + By * self.Sy + Bz * self.Sz

        def spin_hamiltonian_mol_basis(self, D, E, B, theta, phi) : 
                Bz = B * math.cos(theta) 
                Bx = B * math.sin(theta) * math.cos(phi) 
                By = B * math.sin(theta) * math.sin(phi) 

                return self.fine_structure(D, E) + self.zeeman(Bx, By, Bz)

        def spin_hamiltonian_field_basis(self, D, E, B, theta, phi) : 
                return self.fine_structure(D, E, Rotatino(0, -theta, -phi+math.pi/2.)) + self.zeeman(0, 0, B)

        def eval(self, D, E, B, theta = 0, phi = 0, mol_basis = True) : 
                if mol_basis: 
                        return np.linalg.eigvalsh(self.spin_hamiltonian_mol_basis(D, E, B, theta, phi))
                else: 
                        return np.linalg.eigvalsh(self.spin_hamiltonian_field_basis(D, E, B, theta, phi))
                


class TwoTriplets :
        def __init__ (self) :
                self.triplet = TripletHamiltonian()
                self.E = None 
                self.D = None 
                self.J = None
                self.Jdip = None
                self.B = None
                self.matrix_size = 9

                s2i3 = math.sqrt(2.0/3.0);
                si2 = 1.0/math.sqrt(2.0);
                si3 = 1.0/math.sqrt(3.0);
                si6 = 1.0/math.sqrt(6.0);
                
                self.Jproj = np.array( [ [ 0, 0, si3, 0, -si3, 0, si3, 0, 0 ],
                                        [ 0, 0, 0, 0, 0, -si2, 0, si2, 0 ],
                                        [ 0, 0, -si2, 0, 0, 0, si2, 0, 0 ],
                                        [ 0, -si2, 0, si2, 0, 0, 0, 0, 0 ],
                                        [ 0,  0, 0,   0, 0, 0, 0, 0, 1.0 ],
                                        [ 0, 0, 0, 0, 0, si2, 0, si2, 0 ],
                                        [ 0, 0, si6, 0, s2i3, 0, si6, 0, 0 ], 
                                        [ 0, si2, 0, si2, 0, 0, 0, 0, 0 ],
                                        [ 1.0, 0, 0, 0, 0, 0, 0, 0, 0 ] ] )

        def exchange_matrix(self) : 
                return np.kron(self.triplet.Sx, self.triplet.Sx) + np.kron(self.triplet.Sy, self.triplet.Sy) + np.kron(self.triplet.Sz, self.triplet.Sz)

        
        def dipole_dipole_matrix(self, uvec) : 
            """
            returns dipole-dipole interaction matrix, assumes that uvec an unit normalized 3d vector
            """
            unorm = np.linalg.norm(uvec)
            uvec = uvec / unorm 
            uS = uvec[0] * self.triplet.Sx + uvec[1] * self.triplet.Sy + uvec[2] * self.triplet.Sz
            return (self.exchange_matrix() - 3. * np.kron(uS, uS))

        def load_field_basis_Hamiltonian(self, triplet1_rotation, triplet2_rotation, dip_vec = None) : 
            H1 = self.triplet.fine_structure(self.D, self.E, triplet1_rotation) + self.triplet.zeeman(0, 0, self.B)
            H2 = self.triplet.fine_structure(self.D, self.E, triplet2_rotation) + self.triplet.zeeman(0, 0, self.B)
            self.Hfull = np.kron(H1, self.triplet.Id) + np.kron(self.triplet.Id, H2) + self.J * self.exchange_matrix() 
            if dip_vec is not None:
                self.Hfull += self.Jdip * self.dipole_dipole_matrix(dip_vec)
        
        def diag(self) :
                self.eval,self.evec = np.linalg.eigh(self.Hfull)

        def quintet_content(self, i): 
            iProj = np.dot( self.Jproj[4:9, 0:9], self.evec[0:9, i:i+1] );
            norm2 = np.dot( np.matrix.getH(iProj), iProj );
            return norm2[0,0].real;

        def triplet_content(self, i): 
            iProj = np.dot( self.Jproj[1:4, 0:9], self.evec[0:9, i:i+1] );
            norm2 = np.dot( np.matrix.getH(iProj), iProj );
            return norm2[0,0].real;

        def singlet_content(self, i): 
            iProj = np.dot( self.Jproj[0:1, 0:9], self.evec[0:9, i:i+1] );
            norm2 = np.dot( np.matrix.getH(iProj), iProj );
            return norm2[0,0].real;

  
        def sz_elem(self, i): 
            Sz2 =np.kron(self.triplet.Sz, self.triplet.Id) + np.kron(self.triplet.Id, self.triplet.Sz)
            vi = self.evec[:,i]
            Sz2ii = reduce(np.dot, [ np.matrix.getH(vi), Sz2, vi ])
            return Sz2ii[0,0].real 

        def singlet_projector(self):
            singlet_state = np.asmatrix(self.Jproj[0:1,:])
            return np.dot( np.matrix.getH(singlet_state), singlet_state )


        def Bac_field_basis_matrix(self): 
           return np.kron(self.triplet.Sx, self.triplet.Id) + np.kron(self.triplet.Id, self.triplet.Sx)

        def print_info(self) : 
            print("# D %g" % self.D)
            print("# E %g" % self.E)
            print("# B %g" % self.B)
            print("# J %g" % self.J)
            print("# Jip %g" % self.Jdip)


class ODMR_Signal : 
    """ 
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
    """
    def __init__(self, spin_system) : 
        self.spins = spin_system 
        self.rho0 = np.empty(self.spins.matrix_size, dtype=float)
        self.rho2 = np.empty([self.spins.matrix_size, self.spins.matrix_size], dtype=np.complex_) 
        self.gamma = None
        self.gamma_diag = None

    def update_from_spin_hamiltonian(self) : 
        self.Sproj_eig_basis = reduce(np.dot, [ np.matrix.getH( self.spins.evec ), self.spins.singlet_projector(), self.spins.evec])
        self.V = reduce(np.dot, [ np.matrix.getH( self.spins.evec ), self.spins.Bac_field_basis_matrix(), self.spins.evec ])
        
    def omega_nm(self, n, m) :
        return self.spins.eval[n] - self.spins.eval[m]

    def load_rho0_thermal(self, Temp):  
        sum = 0
        for i in range(self.spins.matrix_size) : 
            rho0_i = math.exp(- self.spins.eval[i] / Temp)
            self.rho0[i] = rho_i
            sum += rho_i
        self.rho0 /= sum

    def load_rho0_from_singlet(self) : 
        sum = 0
        for i in range(self.spins.matrix_size) : 
            self.rho0[i] = self.Sproj_eig_basis[i, i].real
            sum += self.rho0[i]
        self.rho0 /= sum

    def chi1(self, omega):
       c1 = 0j
       for m in range(self.spins.matrix_size): 
           for n in range(self.spins.matrix_size): 	     
               # the contribution to chi1 vanishes for n == m, whether gamma is the same for diagonal and non diagonal elements is not relvant here 
               Vmn = self.V[m, n]
               Vmn_abs2 = Vmn.real * Vmn.real + Vmn.imag * Vmn.imag
               c1 -= (self.rho0[m] - self.rho0[n]) * Vmn_abs2 / ( self.omega_nm(n, m) - omega - 1j * self.gamma );
       return c1


    def find_rho2_explicit(self, omega) :
        for m in range(self.spins.matrix_size): 
            for n in range(self.spins.matrix_size):
                rrr = 0j
                for nu in range(self.spins.matrix_size): 
                    for p in [-1., 1.]: 
                        gamma_nm = self.gamma_diag if m == n else self.gamma
                        rrr += (self.rho0[m] - self.rho0[nu]) * self.V[n, nu] * self.V[nu, m] / ( ( self.omega_nm(n, m) - 1j * gamma_nm ) * ( self.omega_nm(nu, m) - omega * p - 1j * self.gamma ) )
                        rrr -= (self.rho0[nu] - self.rho0[n]) * self.V[n, nu] * self.V[nu, m] / ( ( self.omega_nm(n, m) - 1j * gamma_nm ) * ( self.omega_nm(n, nu) - omega * p - 1j * self.gamma ) )
                self.rho2[n, m] = rrr


    def find_rho2(self, omega):
        Vtmp = np.zeros( (self.spins.matrix_size, self.spins.matrix_size), dtype=np.complex_) 
        for m in range(self.spins.matrix_size):
            for nu in range(self.spins.matrix_size):
                for p in [-1., 1.]:
                    Vtmp[nu, m] += (self.rho0[m] - self.rho0[nu]) * self.V[nu, m] / (self.omega_nm(nu, m) - omega * p - 1j * self.gamma)
        self.rho2 = np.dot(self.V, Vtmp) - np.dot(Vtmp, self.V)
        for m in range(self.spins.matrix_size):
            for n in range(self.spins.matrix_size):
                gamma_nm = self.gamma_diag if m == n else self.gamma
                self.rho2[n, m] /= ( self.omega_nm(n, m) - 1j * gamma_nm );


    def odmr(self, omega):
       odmr_amp = 0j
       self.find_rho2(omega)
       
       for m in range(self.spins.matrix_size):
           for n in range(self.spins.matrix_size):
               odmr_amp += self.rho2[m , n] * self.Sproj_eig_basis[n, m]

       return odmr_amp.real

    
    


def main():
        triplet_pair = TwoTriplets()
        triplet_pair.D = 1
        triplet_pair.E = 0.01
        triplet_pair.J = 0.0
        triplet_pair.B = 5
        triplet_pair.Jdip = 0.1
        Nsamples = 5000
        triplet_pair.print_info()

        np.random.seed(1)

        quintet_max = 0.
        for count in range(Nsamples):
            V1 = 2. * math.pi * np.random.rand(3)
            V2 = 2. * math.pi * np.random.rand(3)
            Ur = random_unit_vector()
            triplet_pair.load_field_basis_Hamiltonian( Rotation(V1), Rotation(V2), Ur )
            triplet_pair.diag()
            si = 0 
            for i in range(0,9):
                si += math.pow(triplet_pair.quintet_content(i), 4.0)

            if si > quintet_max:
                quintet_max = si
                quintet_angles1 = V1
                quintet_angles2 = V2
                quintet_rdip = Ur

        print("# quintet_max " + str(quintet_max))
        print("# Euler angles ")
        print("%g   %g   %g" % ( quintet_angles1[0], quintet_angles1[1], quintet_angles1[2] ) )
        print("%g   %g   %g" % ( quintet_angles2[0], quintet_angles2[1], quintet_angles2[2] ) )
        print("# Rdip ")
        print("%g %g %g" % ( quintet_rdip[0], quintet_rdip[1], quintet_rdip[2] ) )
        triplet_pair.load_field_basis_Hamiltonian( Rotation(quintet_angles1), Rotation(quintet_angles2), quintet_rdip )    
        triplet_pair.diag()
        print("# qunitet/triplet projections at B = " + str(triplet_pair.B))
        for i in range(0,9): 
            print("%g   %g   %g   %g" % ( triplet_pair.quintet_content(i), triplet_pair.triplet_content(i), triplet_pair.singlet_content(i), triplet_pair.sz_elem(i) ) )

        B_span = np.arange(0.0, 2.0 * triplet_pair.B, 1e-3 * triplet_pair.B)
        chi_B = [ 0j ] * len(B_span)
        odmr_B = [ 0 ] * len(B_span)
        N_average = 10

        for sample in range(N_average):
            rot_for_sample = Rotation( 2. * math.pi * np.random.rand(3) )
            R1 = np.dot( rot_for_sample.matrix(), Rotation(quintet_angles1).matrix() )
            R2 = np.dot( rot_for_sample.matrix(), Rotation(quintet_angles2).matrix() ) 
            Rdip = np.dot( rot_for_sample.matrix(), quintet_rdip )
            triplet_pair.load_field_basis_Hamiltonian( Rotation(R1), Rotation(R2), Rdip )    
            triplet_pair.diag()

            odmr_from_triplets = ODMR_Signal(triplet_pair)
            odmr_from_triplets.update_from_spin_hamiltonian()
            odmr_from_triplets.load_rho0_from_singlet()
            odmr_from_triplets.gamma = 1e-2
            odmr_from_triplets.gamma_diag = 1e-2
        
            for count, omega in enumerate(B_span):
                chi_B[count] += odmr_from_triplets.chi1(omega)/N_average
                odmr_B[count] += odmr_from_triplets.odmr(omega)/N_average 

        for count, omega in enumerate(B_span):
            sys.stderr.write("%g   %g   %g   %g\n" % ( omega, chi_B[count].real, chi_B[count].imag, odmr_B[count] ))
        

main()
     	
