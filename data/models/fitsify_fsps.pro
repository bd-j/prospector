
leadin=10.^5.8 ;padding that was added to the SFH of each bin in SFH.dat 
files = file_search('./fsps_output/*spec')
nage = 189
nwave = 1963
for i = 0, n_elements(files) - 1 do begin
   cb=bdj_read_fsps(files[i],extras=cbex) 
   age = (cb.age - leadin)/1e9
   good = where(age GE 0, nage)
   
   outstruct = {age:fltarr(nage+1), sfr:fltarr(nage+1), mstar:fltarr(nage+1), $
                wavelength:fltarr(nwave), f_lambda:fltarr(nwave, nage+1), $
                filename:'', units:'Gyr, Msun/yr, Msun, AA, and L_sun/AA'}

   
   outstruct.sfr = [cbex.sfr_yr[0],cbex.sfr_yr]
   outstruct.mstar = [0,cbex.m_]
   outstruct.wavelength = cb.wave
   outstruct.f_lambda = cb.flux[
   outstruct.filename = files[i]
   mwrfits,outstruct,repstr(repstr(files[i],'.spec','.fits'),'fsps_output/',''),/create

endfor
end
