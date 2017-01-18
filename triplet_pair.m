clear, clf;
Exp.Harmonic = 0;
Exp.Field = 0; 
Exp.mwRange = [0 2];
% Exp.mwFreq = 9.5;
% Exp.Range = [280 400];
Opt.nKnots = 100; % get artefacts if don't sample enough angles

Sys.S = 1;  % single triplet
d = 1; e = 0;
Sys.D = [d e]*1e3
Sys.lw = 10;
% [x1,spc1] = pepper(Sys,Exp,Opt);

Sys.S = [1 1] % triplet pair
Sys.D = [Sys.D; Sys.D];
Sys.DFrame = [0 0 0; 0 0 0];
J = 1; d2 = 100; e2 = 0;
Sys.ee = J + [1+e2 1-e2 -2]*d2; % dipolar triplet-triplet coupling, in MHz
Sys.eeFrame = [0 0 0];
[x2,spc2] = pepper(Sys,Exp,Opt);
% plot(x1, spc1/max(spc1),x2,spc2/max(spc2));

% fid=fopen('/home/sam/research/odmr/ttcNov16/test.dat'); expdata=fscanf(fid,'%g',[6 inf]); fclose(fid);
% spcExp = abs(expdata(2,:)) / max(abs(expdata(2,:))); fExp = expdata(1,:)/1e9;
% fid=fopen('/home/sam/research/odmr/ttcNov21_dropcast/sweepFreq550LP0mT331Hz12dBm0p96A161123v.avg'); expdata=fscanf(fid,'%g',[7 inf]); fclose(fid);
% spcExp = expdata(2,:) / max(abs(expdata(2,:))); fExp = expdata(1,:)/1e9;
plot(x2, spc2/max(spc2))

% Vary.D = [1 1; 1 1]*0.1*1e3;
% Vary.ee = [1 1 1]*5;
%Vary.eeFrame = [0 pi/2 0];
% FitOpt.Method = 'genetic fcn';   % genetic algorithm with spectra as is
% FitOpt.Scaling = 'maxabs';    % scaling so that the maximum absolute values coincide

% esfit('mymy', spcExp, Sys, Vary, Exp, Opt);


