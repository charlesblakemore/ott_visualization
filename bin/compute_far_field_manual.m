
datapath = '../raw_data/';

%%% Position of the microsphere relative to the beam
xOffset = 2.0e-6;
yOffset = 0.0e-6;
zOffset = 2.0e-6;

c = 2.998e8;

%%% Bangs 5um spheres
% mbead = 84.0e-15;
% radius = 2.35e-6;
% n_particle = 1.33;

%%% German 7.5um spheres
mbead = 420.0e-15;
radius = 3.76e-6;
n_particle = 1.39;

%%% Vacuum
n_medium = 1.0;

%%% Standard OTT definitions
wavelength0 = 1064.0e-9;
wavelength_medium = wavelength0 / n_medium;

%%% Old trap
% NA = 0.12;

%%% New trap
NA = 0.095;

%%% Stick to linear x-polarization for consistency
polarisation = [1 0];

%%% Max number of terms
Nmax = 100;

saveFormatSpec = 'r%0.2fum_n%0.2f_na%0.3f_x%0.2f_y%0.2f_z%0.2f';
saveName = strrep(sprintf(saveFormatSpec, radius*1e6, n_particle, NA, ...
                          xOffset*1e6, yOffset*1e6, zOffset*1e6), '.', '_');
saveFarfield = true;

%%% number of points to sample when saving result
ntheta = 200;
nphi = 200;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Define the scatterer
T = ott.TmatrixMie(radius, 'wavelength0', wavelength0, ...
                   'index_medium', n_medium, ...
                   'index_particle', n_particle, ...
                   'Nmax', Nmax);
               
%%% Unique Tmatrix for internal fields             
Tint = ott.TmatrixMie(radius, 'wavelength0', wavelength0, ...
                      'index_medium', n_medium, ...
                      'index_particle', n_particle, ...
                      'internal', true, 'Nmax', Nmax);
                   
%%% Construct the input beam
ibeam = ott.BscPmGauss('NA', NA, 'polarisation', polarisation, ...
                       'index_medium', n_medium, ...
                       'wavelength0', wavelength0, ...
                       'Nmax', Nmax);

ibeam = ibeam.translateXyz([xOffset; yOffset; -zOffset], 'Nmax', Nmax);

%%% LET THE SCATTERING BEGIN %%%
sbeam = T * ibeam;
intbeam = Tint * ibeam;

%%% Construct a "total" representation of the beam
totbeam = sbeam.totalField(ibeam);






if saveFarfield

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%     FAR FIELD     %%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Intent is to visualize forward- and back-scattered light patterns to 
    %%% better inform design of the imaging system, as well as operating point
    %%% of MS within a trap

    %%% Construct the points we want to sample over the full unit sphere
    thetapts = linspace(0,pi,ntheta);
    phipts = linspace(0,2*pi,nphi);

    %%% Build up a 2x(ntheta*nphi) array of all sampled points since that's
    %%% how the OTT utility function samples the electric field
    farpts = zeros(2, ntheta*nphi);
    for i = 1:ntheta
        for k = 1:nphi
            ind = (k-1)*ntheta + i;
            farpts(:,ind) = [thetapts(i);phipts(k)];
        end
    end

    %%% Ensure we're using the outgoing basis representation for non-vanishing
    %%% far-fields (since we have to regularize for the singularity at the 
    %%% origin in the actual scattering problem)
    ibeam.basis = 'outgoing';
    sbeam.basis = 'outgoing';
    totbeam.basis = 'outgoing';

    %%% Sample the fieds at the desired points
    [Ei_far, Hi_far] = ibeam.farfield(farpts(1,:),farpts(2,:));
    [Es_far, Hs_far] = sbeam.farfield(farpts(1,:),farpts(2,:));
    [Et_far, Ht_far] = totbeam.farfield(farpts(1,:),farpts(2,:));

    %%% Write all that shit to a few files
    mkdir('../raw_data', saveName)

    formatSpec = '../raw_data/%s/farfield_%s_%s.txt';

    writematrix(farpts, sprintf('../raw_data/%s/farfield_points.txt', saveName));

    writematrix(real(Ei_far), sprintf(formatSpec, saveName, 'inc', 'real'));
    writematrix(imag(Ei_far), sprintf(formatSpec, saveName, 'inc', 'imag'));

    writematrix(real(Es_far), sprintf(formatSpec, saveName, 'scat', 'real'));
    writematrix(imag(Es_far), sprintf(formatSpec, saveName, 'scat', 'imag'));

    writematrix(real(Et_far), sprintf(formatSpec, saveName, 'tot', 'real'));
    writematrix(imag(Et_far), sprintf(formatSpec, saveName, 'tot', 'imag'));

end
