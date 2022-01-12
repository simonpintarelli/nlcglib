#pragma once

#include <cmath>
#include <iomanip>
#include <stdexcept>
#include "utils/logger.hpp"

namespace nlcglib {


struct failed_to_converge : std::exception {};

/**
 *  Newton minimization to determine the chemical potential.
 *
 *  \param  N       number of electrons as a function of \f$\mu\f$
 *  \param  dN      \f$\partial_\mu N(\mu)\f$
 *  \param  ddN     \f$\partial^2_\mu N(\mu)\f$
 *  \param  mu0     initial guess
 *  \param  ne      target number of electrons
 *  \param  tol     tolerance
 *  \param  maxstep max number of Newton iterations
 */
template <class Nt, class DNt, class D2Nt>
double
newton_minimization_chemical_potential(Nt&& N, DNt&& dN, D2Nt&& ddN, double mu0, double ne, double tol, int maxstep = 1000)
{
    double mu  = mu0;
    int iter{0};
    while (true) {
        // compute
        double Nf   = N(mu);
        double dNf  = dN(mu);
        double ddNf = ddN(mu);
        /* minimize (N(mu) - ne)^2  */
        // double F = (Nf-ne)*(Nf-ne);
        double dF = 2*(Nf-ne) * dNf;
        double ddF = 2*dNf*dNf + 2*(Nf-ne) * ddNf;
        // std::printf("%d Nf: %.4f, dNf: %.4f, ddF: %.4f, mu: %.4f\n", iter, Nf, dNf, ddF, mu);

        if (std::abs(ddF) < 1e-10) {
          Logger::GetInstance() << "*Warning* Newton minimization failed (2nd deriv~=0) to find the Fermi energy, "
                                   "using bisection search.\n";
          throw failed_to_converge();
          // TERMINATE(s);
        }


        mu = mu - dF / std::abs(ddF);

        if (std::abs(dF) < tol) {
          if (std::abs(N(mu) - ne) > tol) {
            std::cout << "newton got stuck in a flat region" << "\n";
            throw failed_to_converge();
          }
          return mu;
        }

        iter++;
        if (iter > maxstep) {
            std::stringstream s;
            Logger::GetInstance() << "*Warning* Newton minimization failed (maxsteps) to find the Fermi energy, using bisection search.\n";
            std::cout << "Newton failed" << "\n";
            s << "Newton minimization (chemical potential) failed after 10000 steps!\n";
            throw failed_to_converge();
            // TERMINATE(s);
        }
    }
}



}  // nlcglib