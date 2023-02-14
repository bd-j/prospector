#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_nzsfh_prior():
   from prospect.models import priors_beta as pb
   prior = pb.NzSFH(zred_mini=1e-3, zred_maxi=15.0,mass_mini=7.0, mass_maxi=12.5,
                    z_mini=-1.98, z_maxi=0.19, logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                    const_phi=True)
