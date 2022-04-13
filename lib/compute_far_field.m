function [saveName] = compute_far_field(args)
    arguments
        args.datapath string = '../raw_data/'
        args.include_id logical = false
        args.radius double = 2.35e-6
        args.n_particle double = 1.39
        args.n_medium double = 1.00
        args.wavelength double = 1064.0e-9
        args.NA double = 0.12
        args.xOffset double = 0.0e-6
        args.yOffset double = 0.0e-6
        args.zOffset double = 0.0e-6
        args.halfCone double = pi/2.0
        args.ntheta double = 101
        args.nphi double = 101
        args.Nmax double = 50
        args.polarisation string = 'X'
    end

if strcmp(args.datapath, '../raw_data/')
    args.include_id = true;
end

%%% Handle the arguments properly for both internal matlab execution
%%% as well as command-line execution where arguments are necessarily
%%% strings (probably printed by bash or equivalent)

wavelength_medium = args.wavelength / args.n_medium;

if strcmp(args.polarisation, 'X')
    polarisation = [1 0];
elseif strcmp(args.polarisation, 'Y')
    polarisation = [0 1];
else
    polarisation = [1 0];
end

saveFormatSpec = 'r%0.2fum_n%0.2f_na%0.3f_x%0.2f_y%0.2f_z%0.2f_Nmax%i';
saveName = strrep(sprintf(saveFormatSpec, args.radius*1e6, ...
                          args.n_particle, args.NA, ...
                          args.xOffset*1e6, args.yOffset*1e6, ...
                          args.zOffset*1e6, args.Nmax), '.', '_');

if args.include_id
    saveName = strcat(strip(args.datapath,'right','/'), '/', saveName);
else
    saveName = args.datapath;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Define the scatterer
T = ott.TmatrixMie(args.radius, 'wavelength0', args.wavelength, ...
                   'index_medium', args.n_medium, ...
                   'index_particle', args.n_particle, ...
                   'Nmax', args.Nmax);
               
%%% Unique Tmatrix for internal fields             
Tint = ott.TmatrixMie(args.radius, 'wavelength0', args.wavelength, ...
                      'index_medium', args.n_medium, ...
                      'index_particle', args.n_particle, ...
                      'internal', true, 'Nmax', args.Nmax);
                   
%%% Construct the input beam
ibeam = ott.BscPmGauss('NA', args.NA, 'polarisation', polarisation, ...
                       'index_medium', args.n_medium, ...
                       'wavelength0', args.wavelength, ...
                       'Nmax', args.Nmax);

ibeam = ibeam.translateXyz([args.xOffset; args.yOffset; -args.zOffset], ...
                           'Nmax', args.Nmax);

%%% LET THE SCATTERING BEGIN %%%
sbeam = T * ibeam;
intbeam = Tint * ibeam;

%%% Construct a "total" representation of the beam
totbeam = sbeam.totalField(ibeam);


%%% Translate back to the origin after doing the scattering, since the 
%%% farfield imaging is aligned to the optical focus, not the scatterer
% ibeam = ibeam.translateXyz([-args.xOffset; -args.yOffset; +args.zOffset], ...
%                             'Nmax', args.Nmax);
% sbeam = sbeam.translateXyz([-args.xOffset; -args.yOffset; +args.zOffset], ...
%                             'Nmax', args.Nmax);
% totbeam = totbeam.translateXyz([-args.xOffset; -args.yOffset; +args.zOffset], ...
%                                 'Nmax', args.Nmax);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%     FAR FIELD     %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Intent is to visualize forward- and back-scattered light patterns to 
%%% better inform design of the imaging system, as well as operating point
%%% of MS within a trap

%%% Construct the points we want to sample over the full unit sphere
thetapts_trans = linspace(0, args.halfCone, args.ntheta);
thetapts_refl = linspace(pi-args.halfCone, pi, args.ntheta);
phipts = linspace(0, 2*pi, args.nphi);

%%% Build up a 2x(ntheta*nphi) array of all sampled points since that's
%%% how the OTT utility function samples the electric field
farpts_trans = zeros(2, args.ntheta*args.nphi);
farpts_refl = zeros(2, args.ntheta*args.nphi);
for i = 1:args.ntheta
    for k = 1:args.nphi
        ind = (k-1)*args.ntheta + i;
        farpts_trans(:,ind) = [thetapts_trans(i);phipts(k)];
        farpts_refl(:,ind) = [thetapts_refl(i);phipts(k)];
    end
end


%%% Ensure we're using the outgoing basis representation for non-vanishing
%%% far-fields (since we have to regularize for the singularity at the 
%%% origin in the actual scattering problem)
ibeam.basis = 'outgoing';
sbeam.basis = 'outgoing';
totbeam.basis = 'outgoing';


%%% Sample the fieds at the desired points
[Ei_trans, Hi_trans] = ibeam.farfield(farpts_trans(1,:),farpts_trans(2,:));
[Es_trans, Hs_trans] = sbeam.farfield(farpts_trans(1,:),farpts_trans(2,:));
[Et_trans, Ht_trans] = totbeam.farfield(farpts_trans(1,:),farpts_trans(2,:));

[Ei_refl, Hi_refl] = ibeam.farfield(farpts_refl(1,:),farpts_refl(2,:));
[Es_refl, Hs_refl] = sbeam.farfield(farpts_refl(1,:),farpts_refl(2,:));
[Et_refl, Ht_refl] = totbeam.farfield(farpts_refl(1,:),farpts_refl(2,:));

%%% Write all that shit to a few files
mkdir(saveName);
disp(' ')
disp('Writing data to:');
disp(sprintf('    %s\n', saveName));

formatSpec = '%s/farfield_%s_%s_%s.txt';

writematrix(farpts_trans, ...
            sprintf('%s/farfield_points_trans.txt', saveName));

writematrix(real(Ei_trans), sprintf(formatSpec, saveName, ...
                                    'inc', 'trans', 'real'));
writematrix(imag(Ei_trans), sprintf(formatSpec, saveName, ...
                                    'inc', 'trans', 'imag'));

writematrix(real(Es_trans), sprintf(formatSpec, saveName, ...
                                    'scat', 'trans', 'real'));
writematrix(imag(Es_trans), sprintf(formatSpec, saveName, ...
                                    'scat', 'trans', 'imag'));

writematrix(real(Et_trans), sprintf(formatSpec, saveName, ...
                                    'tot', 'trans', 'real'));
writematrix(imag(Et_trans), sprintf(formatSpec, saveName, ...
                                    'tot', 'trans', 'imag'));


writematrix(farpts_refl, ...
            sprintf('%s/farfield_points_refl.txt', saveName));

writematrix(real(Ei_refl), sprintf(formatSpec, saveName, ...
                                   'inc', 'refl', 'real'));
writematrix(imag(Ei_refl), sprintf(formatSpec, saveName, ...
                                   'inc', 'refl', 'imag'));

writematrix(real(Es_refl), sprintf(formatSpec, saveName, ...
                                   'scat', 'refl', 'real'));
writematrix(imag(Es_refl), sprintf(formatSpec, saveName, ...
                                   'scat', 'refl', 'imag'));

writematrix(real(Et_refl), sprintf(formatSpec, saveName, ...
                                   'tot', 'refl', 'real'));
writematrix(imag(Et_refl), sprintf(formatSpec, saveName, ...
                                  'tot', 'refl', 'imag'));


end
