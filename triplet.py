
import numpy as np
from numpy.linalg import eig
from pylab import *
import cmath 
from math import *
from scipy.optimize import fsolve, brentq
from scipy.optimize import minimize_scalar
import sys


def random_unit_vector() : 
    phi = 2.0 * math.pi * np.random.random()
    z = 2.0 * np.random.random() - 1.0
    r = math.sqrt(1.0 - z*z)
    return np.array([r * math.cos(phi), r * math.sin(phi), z ])


class TripletHamiltonian : 
        def __init__ (self) :
                self.Id = np.matrix('1 0 0; 0 1 0; 0 0 1', dtype=np.complex_)
                self.Sz = np.matrix('1 0 0; 0 0 0; 0 0 -1', dtype=np.complex_)
                self.Sx = np.matrix('0 1 0; 1 0 1; 0 1 0', dtype=np.complex_) / math.sqrt(2.0)
                self.Sy = - 1j * np.matrix('0 1 0; -1 0 1; 0 -1 0', dtype=np.complex_) / math.sqrt(2.0)

        def euler2mat_z1x2z3(self, z1 = 0, x2 = 0, z3 = 0) :
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
        
        
        def fine_structure(self, D, E, z1=0, x2=0, z3=0) :
                rotation_matrix = self.euler2mat_z1x2z3(z1, x2, z3)
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
                return self.fine_structure(D, E, 0, -theta, -phi+math.pi/2.) + self.zeeman(0, 0, B)

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

        def load_field_basis_Hamiltonian(self, triplet1_angles, triplet2_angles, dip_vec = None) : 
            H1 = self.triplet.fine_structure(self.D, self.E, triplet1_angles[0], triplet1_angles[1], triplet1_angles[2]) + self.triplet.zeeman(0, 0, self.B)
            H2 = self.triplet.fine_structure(self.D, self.E, triplet2_angles[0], triplet2_angles[1], triplet2_angles[2]) + self.triplet.zeeman(0, 0, self.B)
            self.Hfull = np.kron(H1, self.triplet.Id) + np.kron(self.triplet.Id, H2) + self.J * self.exchange_matrix() 
            if dip_vec is not None:
                self.Hfull += self.Jdip * self.dipole_dipole_matrix(dip_vec)
        
        def diag(self) :
                self.Heval,self.Hevec = np.linalg.eigh(self.Hfull)

        def quintet_content(self, i): 
            iProj = np.dot( self.Jproj[4:9, 0:9], self.Hevec[0:9, i:i+1] );
            norm2 = np.dot( np.matrix.getH(iProj), iProj );
            return norm2[0,0].real;

        def triplet_content(self, i): 
            iProj = np.dot( self.Jproj[1:4, 0:9], self.Hevec[0:9, i:i+1] );
            norm2 = np.dot( np.matrix.getH(iProj), iProj );
            return norm2[0,0].real;

        def singlet_content(self, i): 
            iProj = np.dot( self.Jproj[0:1, 0:9], self.Hevec[0:9, i:i+1] );
            norm2 = np.dot( np.matrix.getH(iProj), iProj );
            return norm2[0,0].real;

  
        def sz_elem(self, i): 
            Sz2 =np.kron(self.triplet.Sz, self.triplet.Id) + np.kron(self.triplet.Id, self.triplet.Sz)
            vi = self.Hevec[:,i]
            Sz2ii = reduce(np.dot, [ np.matrix.getH(vi), Sz2, vi ])
            return Sz2ii[0,0].real 

        def size(self): 
            return 9

        def print_info(self) : 
            print("# D %g" % self.D)
            print("# E %g" % self.E)
            print("# B %g" % self.B)
            print("# J %g" % self.J)
            print("# Jip %g" % self.Jdip)

def main():
        t = TwoTriplets()
        t.D = 1
        t.E = 0.3
        t.J = 0.0
        t.B = 5
        t.Jdip = 0.1
        Nsamples = 5000
        t.print_info()

        np.random.seed(1)

        quintet_max = 0.
        for count in range(Nsamples):
            V1 = 2. * math.pi * np.random.rand(3)
            V2 = 2. * math.pi * np.random.rand(3)
            Ur = random_unit_vector()
            t.load_field_basis_Hamiltonian( V1, V2, Ur )
            t.diag()
            si = 0 
            for i in range(0,9):
                si += math.pow(t.quintet_content(i), 4.0)

            if si > quintet_max:
                quintet_max = si
                quintet_angles1 = V1
                quintet_angles2 = V2
                quintet_rdip = Ur

        # quintet_angles1 = np.array( [ 5.8308, 4.34636, 3.31015 ] )
        # quintet_angles2 = np.array( [ 2.88627, 4.83054, 2.48384 ] )
        # quintet_rdip = np.array( [ 0.0645974, 0.0136964, 0.997817 ] )

        print("# quintet_max " + str(quintet_max))
        print("# Euler angles ")
        print("%g   %g   %g" % ( quintet_angles1[0], quintet_angles1[1], quintet_angles1[2] ) )
        print("%g   %g   %g" % ( quintet_angles2[0], quintet_angles2[1], quintet_angles2[2] ) )
        print("# Rdip ")
        print("%g %g %g" % ( quintet_rdip[0], quintet_rdip[1], quintet_rdip[2] ) )
        t.load_field_basis_Hamiltonian( quintet_angles1, quintet_angles2, quintet_rdip )    
        t.diag()
        print("# qunitet/triplet projections at B = " + str(t.B))
        for i in range(0,9): 
            print("%g   %g   %g   %g" % ( t.quintet_content(i), t.triplet_content(i), t.singlet_content(i), t.sz_elem(i) ) )

main()
		
