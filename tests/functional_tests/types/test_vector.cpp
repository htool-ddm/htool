#include "htool/types/vector.hpp"

using namespace std;
using namespace htool;
int main(int argc, char const *argv[]) {

    bool test = 0;

    //// Vectors
    // int
    vector<int> ai(10, 0);
    vector<int> aiplus  = ai;
    vector<int> aimult  = ai;
    vector<int> aidiv   = ai;
    vector<int> aimult3 = ai;
    vector<int> aidiv3  = ai;
    int iprod           = 0;
    double inorm        = 0;
    int imean           = 0;
    for (int i = 0; i < ai.size(); i++) {
        ai[i]     = i;
        aiplus[i] = 2 * i;
        aidiv[i]  = i / 3;
        iprod += i * i;
        imean += i;
        aimult3[i] = 3 * i;
        aidiv3[i]  = i;
    }
    inorm = sqrt(iprod);
    imean /= ai.size();
    vector<int> bi = ai;

    cout << "ai = " << ai << endl;
    test = test || !(ai == bi);
    cout << "bi = " << bi << endl;
    test = test || !(ai + bi == aiplus);
    cout << "ai+bi       = " << ai + bi << endl;
    test = test || !((ai - bi) == vector<int>(ai.size(), 0));
    cout << "ai-bi       = " << ai - bi << endl;
    test = test || !(ai / 3 == aidiv);
    cout << "ai/3        = " << ai / 3 << endl;
    test = test || !(iprod == dprod(ai, bi));
    cout << "dprod(ai,bi)= " << dprod(ai, bi) << endl;
    test = test || !((inorm - norm2(ai)) < 1e-16);
    cout << "norm2(ai)   = " << norm2(ai) << endl;
    test = test || !(argmax(ai) == ai.size() - 1);
    cout << "argmax(ai)  = " << argmax(ai) << endl;
    test = test || !(abs(max(ai + bi) - aiplus[aiplus.size() - 1]) < 1e-16);
    cout << "max(ai+bi)  = " << max(ai + bi) << endl;
    test = test || !(imean == mean(ai));
    cout << "mean(ai)    = " << mean(ai) << endl;
    ai *= 3;
    test = test || !(ai == aimult3);
    cout << "ai*=3    ai = " << ai << endl;
    ai /= 3;
    test = test || !(ai == ai);
    cout << "ai/=3    ai = " << ai << endl;

    // double
    vector<double> ad(10, 0);
    vector<double> adplus  = ad;
    vector<double> admult  = ad;
    vector<double> addiv   = ad;
    vector<double> admult3 = ad;
    vector<double> addiv3  = ad;
    double ddprod          = 0;
    double dnorm           = 0;
    double dmean           = 0;
    for (int i = 0; i < ad.size(); i++) {
        ad[i]     = i;
        adplus[i] = 2 * i;
        addiv[i]  = i / 3.;
        ddprod += i * i;
        dmean += i;
        admult3[i] = 3. * i;
        addiv3[i]  = i;
    }
    dnorm = sqrt(ddprod);
    dmean /= ad.size();
    vector<double> bd = ad;

    cout << "bd = " << bd << endl;
    test = test || !(norm2(ad + bd - adplus) < 1e-16);
    cout << "ad+bd       = " << ad + bd << endl;
    test = test || !(norm2(ad - bd) < 1e-16);
    cout << "ad-bd       = " << ad - bd << endl;
    test = test || !(norm2(ad / 3. - addiv) < 1e-16);
    cout << norm2(ad / 3 - addiv) << endl;
    cout << "ad/3        = " << ad / 3. << endl;
    test = test || !(abs(ddprod - dprod(ad, bd)) < 1e-16);
    cout << "dprod(ad,bd)= " << dprod(ad, bd) << endl;
    test = test || !((dnorm - norm2(ad)) < 1e-16);
    cout << "norm2(ad)   = " << norm2(ad) << endl;
    test = test || !(argmax(ad) == ad.size() - 1);
    cout << "argmax(ad)  = " << argmax(ad) << endl;
    test = test || !(abs(max(ad + bd) - adplus[adplus.size() - 1]) < 1e-16);
    cout << "max(ad+bd)  = " << max(ad + bd) << endl;
    test = test || !(abs(dmean - mean(ad)) < 1e-16);
    cout << "mean(ad)    = " << mean(ad) << endl;
    ad *= 3;
    test = test || !(norm2(ad - admult3) < 1e-16);
    cout << "ad*=3    ad = " << ad << endl;
    ad /= 3;
    test = test || !(norm2(ad - addiv3) < 1e-16);
    cout << "ad/=3    ad = " << ad << endl;

    // complex double
    vector<complex<double>> acd(10, 0);
    vector<complex<double>> acdplus  = acd;
    vector<complex<double>> acdmult  = acd;
    vector<complex<double>> acddiv   = acd;
    vector<complex<double>> acdmult3 = acd;
    vector<complex<double>> acddiv3  = acd;
    complex<double> cddprod          = 0;
    double cdnorm                    = 0;
    complex<double> cdmean           = 0;
    for (int i = 0; i < acd.size(); i++) {
        acd[i]     = complex<double>(i, 1);
        acdplus[i] = acd[i] + acd[i];
        acddiv[i]  = acd[i] / 3.;
        cddprod += acd[i] * conj(acd[i]);
        cdmean += acd[i];
        acdmult3[i] = 3. * acd[i];
        acddiv3[i]  = acd[i];
    }
    cdnorm = sqrt(abs(cddprod));
    cdmean /= acd.size();
    vector<complex<double>> bcd = acd;

    cout << "bcd = " << bcd << endl;
    test = test || !(norm2(acd + bcd - acdplus) < 1e-16);
    cout << "acd+bcd       = " << acd + bcd << endl;
    test = test || !(norm2(acd - bcd) < 1e-16);
    cout << "acd-bcd       = " << acd - bcd << endl;
    test = test || !(norm2(acd / 3. - acddiv) < 1e-16);
    cout << "acd/3        = " << acd / 3. << endl;
    test = test || !(abs(cddprod - dprod(acd, bcd)) < 1e-16);
    cout << "dprod(acd,bcd)= " << dprod(acd, bcd) << endl;
    test = test || !((cdnorm - norm2(acd)) < 1e-16);
    cout << "norm2(acd)   = " << norm2(acd) << endl;
    test = test || !(argmax(acd) == acd.size() - 1);
    cout << "argmax(acd)  = " << argmax(acd) << endl;
    test = test || !(abs(max(acd + bcd) - acdplus[acdplus.size() - 1]) < 1e-16);
    cout << "max(acd+bcd)  = " << max(acd + bcd) << endl;
    test = test || !(abs(cdmean - mean(acd)) < 1e-16);
    cout << "mean(acd)    = " << mean(acd) << endl;
    acd *= 3.;
    test = test || !(norm2(acd - acdmult3) < 1e-16);
    cout << "acd*=3    acd = " << acd << endl;
    acd /= 3.;
    test = test || !(norm2(acd - acddiv3) < 1e-16);
    cout << "acd/=3    acd = " << acd << endl;

    return test;
}
