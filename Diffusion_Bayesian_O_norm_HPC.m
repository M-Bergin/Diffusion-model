
%% Parameters setup

%Parameters for numerical model
N_x=151;                %Number of x points
L_size=300;             %Length of simulation domain
delta_t=100e-3;         %Time step to use
T_time=300;             %Total time to model over
flip=0;                 %Option to flip which half of model is high or low



%Setup for importing data
files=[2,3,4];
N_files=length(files);

%Temperatures of experimental data
T=[50,100,150];
T_n={'l','m','h'};

%Initialise some extra variables
N_uplot=200;
u_plot=linspace(0,1,N_uplot);
% D_fit=NaN*zeros(N_uplot,4,N_files);
% D_fit_error=NaN*zeros(N_uplot,4,N_files);
ion_text={'S','C','CF','O'};
fit_output{N_files}=[];


%% Main loop over the three data sets
for n=1:3
    %% Setup the Import Options and import the data (NOT USED NOW)
    
    %     filename=['GS054-',num2str(files(n)),'_p1_C60_1'];
    %     opts = spreadsheetImportOptions("NumVariables", 7);
    %
    %     % Specify sheet and range
    %     opts.Sheet = [filename ,'_decimal place'];
    %     opts.DataRange = "A6:G400";
    %
    %     % Specify column names and types
    %     opts.VariableNames = ["t_s", "C1", "O1", "Si1", "CF1", "S1", "In1"];
    %     opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double"];
    %
    %     % Import the data
    %     tbl = readtable(['C:\Users\mab679\Dropbox\Newcastle\Diffusion model\Data\',filename,'.xlsx'], opts, "UseExcel", false);
    %
    %     %% Convert to output type
    %     t_s = tbl.t_s;
    %     C = tbl.C1;
    %     O = tbl.O1;
    %     Si = tbl.Si1;
    %     CF = tbl.CF1;
    %     S = tbl.S1;
    %     In = tbl.In1;
    %
    %     %Clear the NaNs
    %     nan_ind=isnan(t_s);
    %     t_s(nan_ind)=[];
    %     C(nan_ind)=[];
    %     O(nan_ind)=[];
    %     Si(nan_ind)=[];
    %     CF(nan_ind)=[];
    %     S(nan_ind)=[];
    %     In(nan_ind)=[];
    %
    %
    %     %% Clear temporary variables
    %     clear opts tbl
    %
    %     %% Calculate the depth
    %
    %     a_n=2.155968;%Nafion sputter rate nm/s
    %     a_p=1.497089;%P3HT sputter rate nm/s
    %     jump_ind1=96;%96; 90 when shifted
    %     jump_ind2=122;%122; 130 when shifted
    %     switch n
    %         case 1
    %             jump_ind=101; %101 originally, 95 shifted
    %         case 2
    %             jump_ind=101; %101 original. 93 when shifted
    %     end
    %
    %
    %     x=NaN*zeros(size(t_s));
    %
    %     if n==1 || n==2
    %         x(1:jump_ind)=t_s(1:jump_ind)*a_n;
    %         x(jump_ind+1:end)=a_p*t_s(jump_ind+1:end)+(x(jump_ind)-a_p*t_s(jump_ind));
    %     elseif n==3
    %         x(1:jump_ind1)=t_s(1:jump_ind1)*a_n;
    %         x(jump_ind1+1:jump_ind2)=((a_n+a_p)/2)*t_s(jump_ind1+1:jump_ind2)+(x(jump_ind1)-((a_n+a_p)/2)*t_s(jump_ind1));
    %         x(jump_ind2+1:end)=a_p*t_s(jump_ind2+1:end)+(x(jump_ind2)-a_p*t_s(jump_ind2));
    %     end
    %
    
    
    
    %% Use the original import
    filename = ['/home/mab679/Diffusion/Data/GS054-',num2str(files(n)),'_p1_C60_1_depth_decimal place.TXT'];
    %filename = ['Data\GS054-',num2str(files(n)),'_p1_C60_1_depth_decimal place.TXT'];
    delimiter = '\t';
    startRow = 6;
    
    %% Format for each line of text:
    %   column1: double (%f)
    %	column2: double (%f)
    %   column3: double (%f)
    %	column4: double (%f)
    %   column5: double (%f)
    %	column6: double (%f)
    %   column7: double (%f)
    % For more information, see the TEXTSCAN documentation.
    formatSpec = '%f%f%f%f%f%f%f%[^\n\r]';
    
    %% Open the text file.
    fileID = fopen(filename,'r');
    
    %% Read columns of data according to the format.
    % This call is based on the structure of the file used to generate this
    % code. If an error occurs for a different file, try regenerating the code
    % from the Import Tool.
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    
    %% Close the text file.
    fclose(fileID);
    
    %% Post processing for unimportable data.
    % No unimportable data rules were applied during the import, so no post
    % processing code is included. To generate code which works for
    % unimportable data, select unimportable cells in a file and regenerate the
    % script.
    
    %% Allocate imported array to column variable names
    x = dataArray{:, 1};
    C = dataArray{:, 2};
    O = dataArray{:, 3};
    Si = dataArray{:, 4};
    CF = dataArray{:, 5};
    S = dataArray{:, 6};
    In = dataArray{:, 7};
    
    
    %% Clear temporary variables
    clearvars filename delimiter startRow formatSpec fileID dataArray ans;
    
    
    
    
    
    
    
    
    %% Clip the data to the relevant edge
    if n==1||n==2
        inds=find(x>250 & x<450);
    elseif n==3
        inds=find(x>250 & x<462);
    end
    x_clip=x(inds);
    C_clip=C(inds);
    O_clip=O(inds);
    Si_clip=Si(inds);
    CF_clip=CF(inds);
    S_clip=S(inds);
    In_clip=In(inds);
    
    
    
    
    
    %% Remove the peak before the interface
    if n==1
        peak_inds=(23:27);
        x_orig=x_clip;
        O_orig=O_clip;
        CF_orig=CF_clip;
        x_clip(peak_inds)=[];
        O_clip(peak_inds)=[];
        CF_clip(peak_inds)=[];
    else
        x_orig=x_clip;
        O_orig=O_clip;
        CF_orig=CF_clip;
    end
    
    %P3HT markers
    S_scaled=S_clip/max(S_clip);
    C_scaled=C_clip/max(C_clip);
    
    %Nafion markers
    CF_scaled=CF_clip/max(CF_clip);
    O_scaled=O_clip/max(O_clip);
    
    
    m=4;
    
    switch m
        case 1
            [xData, yData] = prepareCurveData( x_clip, S_scaled );
            scale_val=max(S_clip);
            %ion_text='S';
        case 2
            [xData, yData] = prepareCurveData( x_clip, C_scaled );
            scale_val=max(C_clip);
            %ion_text='C';
        case 3
            [xData, yData] = prepareCurveData( x_clip, CF_scaled );
            scale_val=max(CF_clip);
            y_orig=CF_orig;
            %ion_text='CF';
        case 4
            [xData, yData] = prepareCurveData( x_clip, O_scaled );
            scale_val=max(O_clip);
            y_orig=O_orig;
            %ion_text='O';
    end
    
    %Set the error as double the expected shot noise
    sigma=2*sqrt(yData/scale_val);
    weight=1./sigma.^2;
    
    %% Do the normal fit first to get best parameters
    
    %Set the fit type
    %ft = fittype( 'a2*Diffusion_numeric_1D(x-x0,a,b,c,u0,0)+c0', 'independent', 'x', 'dependent', 'y' );
    fit_type_input=['a2*Diffusion_numeric_1D_mid(x-x0,@(u_D) a+(c/(sqrt(2*pi)*b))*exp(-(u_D-u0).^2/(2*b.^2)),'...
        ,num2str(flip),',',num2str(delta_t),',',num2str(L_size),',',num2str(N_x),',',num2str(T_time),')+c0'];
    ft = fittype( fit_type_input, 'independent', 'x', 'dependent', 'y' ,'coefficients', {'a','a2','b','c','c0','u0','x0'});
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    if n==1||n==2
        if m==1 || m==2
            opts.StartPoint = [1 1 0.12 0 0.1 0.5 350];
        elseif m==3|| m==4
            opts.StartPoint = [1 1 0.12 0 0 0.5 350];
        end
    elseif n==3
        if m==1 || m==2
            opts.StartPoint = [1 1 0.12 1.5 0.1 0.5 375];
        elseif m==3|| m==4
            opts.StartPoint = [1 1 0.12 1.5 0.1 0.5 375];
        end
    end
    opts.Lower = [0 -Inf 0.05 0 -Inf 0.2 -Inf];
    opts.Upper = [Inf Inf 0.7 Inf Inf 0.8 Inf];
    opts.Weights = weight;
    
    % Fit model to data.
    [fitresult, gof,o] = fit( xData, yData, ft, opts );
    fit_output{n}=fitresult;
    
    
    %% Do bayesian analysis to find the error on best parameters
    
    %Setup range of variables to use
    
    if n==1
        a_vec.(T_n{n})=linspace(0,0.14,80);
        c_vec.(T_n{n})=linspace(0,0.1,40);
        b_vec.(T_n{n})=linspace(0.01,0.4,40);
        u0_vec.(T_n{n})=linspace(0.2,0.8,40);
    elseif n==2
        a_vec.(T_n{n})=linspace(0.12,0.25,80);
        c_vec.(T_n{n})=linspace(0,0.12,40);
        b_vec.(T_n{n})=linspace(0.01,0.25,40);
        u0_vec.(T_n{n})=linspace(0.2,0.8,40);
    elseif n==3
        a_vec.(T_n{n})=linspace(0.6,1.6,80);
        c_vec.(T_n{n})=linspace(1.3,1.8,40);
        b_vec.(T_n{n})=linspace(0.07,0.15,40);
        u0_vec.(T_n{n})=linspace(0.37,0.47,40);
        
    end
    
    %Use fitted parameters for some variables
    a2=fitresult.a2;
    c0=fitresult.c0;
    x0_vec.(T_n{n})=fitresult.x0;
    
    %Initialise some variables
    N_a=length(a_vec.(T_n{n}));
    N_c=length(c_vec.(T_n{n}));
    N_b=length(b_vec.(T_n{n}));
    N_u0=length(u0_vec.(T_n{n}));
    N_x0=length(x0_vec.(T_n{n}));
    
    chi2=NaN*ones(N_a,N_c,N_b,N_u0,N_x0);
    
    
    
    N_data=length(xData);
    y_est=NaN*ones(N_data,N_a);
    
    
    %Set current vectors to reduce communication to parfor workers
    a_vec_T=a_vec.(T_n{n});
    b_vec_T=b_vec.(T_n{n});
    c_vec_T=c_vec.(T_n{n});
    u0_vec_T=u0_vec.(T_n{n});
    x0_vec_T=x0_vec.(T_n{n});
    
    
    %Main loop over all the variables
    parfor n_a=1:N_a
        for n_c=1:N_c
            for n_b=1:N_b
                for n_u0=1:N_u0
                    for n_x0=1:N_x0
                        %Caluculate D for this set of variables
                        D_gen= @(u_D) a_vec_T(n_a)+(c_vec_T(n_c)/(sqrt(2*pi)*b_vec_T(n_b)))*exp(-(u_D-u0_vec_T(n_u0)).^2/(2*b_vec_T(n_b).^2));
                        
                        %Perform diffusion calculation to obtain estimate
                        y_est=a2*Diffusion_numeric_1D_mid(xData-x0_vec_T(n_x0),D_gen,flip,delta_t,L_size,N_x,T_time)+c0;
                        
                        %Old function
                        %y_est=a2*Diffusion_numeric_1D(xData-x0_vec.(T_n{n})(n_x0),a_vec.(T_n{n})(n_a),b_vec.(T_n{n})(n_b),c_vec.(T_n{n})(n_c),u0_vec.(T_n{n})(n_u0),0)+c0;
                        
                        %Find chi^2
                        chi2(n_a,n_c,n_b,n_u0,n_x0)=sum(((yData-y_est)./sigma).^2);
                    end
                end
            end
        end
        
        %Counter display
        %disp([num2str(n_a),'/',num2str(N_a)]);
    end
    y_est_save.((T_n{n}))=y_est;
    chi2_save.((T_n{n}))=chi2;
    L.((T_n{n}))=exp(-chi2/2);%/((2*pi)^(N_data/2)*sqrt(N_data));
    
    
    
    
    %% Generate diffusivity for each of the paramaters
    
    u_test=linspace(0,1,100);
    %u_test=0.6;
    N_utest=length(u_test);
    
    D_mean.(T_n{n})=NaN*zeros(1,N_utest);
    D_std.(T_n{n})=NaN*zeros(1,N_utest);
    D_low.(T_n{n})=NaN*zeros(1,N_utest);
    D_upp.(T_n{n})=NaN*zeros(1,N_utest);
    
    %Loop over each concentration from 0 to 1
    for k_u=1:N_utest
        D.(T_n{n})=NaN*ones(N_a,N_c,N_b,N_u0,N_x0);
        
        %Loop to find each diffusivity, probably could be vectorised to
        %improve speed
        for n_a=1:N_a
            for n_c=1:N_c
                for n_b=1:N_b
                    for n_u0=1:N_u0
                        for n_x0=1:N_x0
                            D.(T_n{n})(n_a,n_c,n_b,n_u0,n_x0)=a_vec.(T_n{n})(n_a)+(c_vec.(T_n{n})(n_c)/(sqrt(2*pi)*b_vec.(T_n{n})(n_b)))*exp(-(u_test(k_u)-u0_vec.(T_n{n})(n_u0)).^2/(2*b_vec.(T_n{n})(n_b)^2));
                        end
                    end
                end
            end
        end
        
        
        D_mean.(T_n{n})(k_u)=sum(D.(T_n{n}).*L.(T_n{n}),[1,2,3,4,5])/sum(L.(T_n{n})(:));
        D_std.(T_n{n})(k_u)=sqrt(sum((D.(T_n{n})-D_mean.(T_n{n})(k_u)).^2.*L.(T_n{n}),[1,2,3,4,5])/sum(L.(T_n{n})(:)));
        
        %[histw,intervals]=histwc(D.(T_n{n})(:), L.(T_n{n})(:), 30);
        %figure;bar(intervals, histw)
        
        
        %Find cumulative probability of diffusivity
        D_v=D.(T_n{n})(:);
        L_v=L.(T_n{n})(:);
        [D_s,idx] = sort(D_v);
        L_cum = cumsum(double(L_v(idx)));
        L_cum=L_cum/L_cum(end);
        %figure;plot(D_s,L_cum)
        
        %Find lower and upper bounds from cumulative probability
        [~,idx_min]=min(abs(L_cum-0.1));
        [~,idx_max]=min(abs(L_cum-0.9));
        
        D_low.(T_n{n})(k_u)=D_s(idx_min);
        D_upp.(T_n{n})(k_u)=D_s(idx_max);
        
        %D_low.(T_n{n})(k_u)=interp1(L_cum([idx_min-1:idx_min+1]),D_s([idx_min-1:idx_min+1]),0.1);
        %D_upp.(T_n{n})(k_u)=interp1(L_cum([idx_max-1:idx_max+1]),D_s([idx_max-1:idx_max+1]),0.9);
    end
    
    
    
    %% Plot fit with data
    
    x_plot=linspace(min(x_clip),max(x_clip),300);
    
    %     figure;subplot(2,1,1)
    %     errorbar(x_clip,yData,sigma,'x','MarkerSize',6,'LineWidth',1)
    %     %plot(x_clip,yData,'x','MarkerSize',6,'LineWidth',1)
    %     hold on
    %     plot(x_plot,fitresult(x_plot),'Linewidth',1)
    %     if n==1
    %         plot(x_orig(peak_inds),y_orig(peak_inds)/scale_val,'rx','MarkerSize',6,'LineWidth',1)
    %     end
    %     plot([fitresult.x0 fitresult.x0],[0 1],'--','LineWidth',1,'Color',[0.4940, 0.1840, 0.5560])
    % %     if n==1 || n==2
    % %         plot([x(jump_ind) x(jump_ind)],[0 1],'k--','LineWidth',1)
    % %     elseif n==3
    % %         plot([x(jump_ind1) x(jump_ind1)],[0 1],'k--','LineWidth',1)
    % %         plot([x(jump_ind2) x(jump_ind2)],[0 1],'k--','LineWidth',1)
    % %     end
    %     xlabel('x/nm')
    %     ylabel('Relative concentration')
    %     title(['Ion: ',ion_text{m},', T=',num2str(T(n))])
    %     set(gca,'FontSize',12,'Linewidth',1)
    %
    %     subplot(2,1,2)
    %
    D_fit.(T_n{n})=fitresult.a+(fitresult.c/(sqrt(2*pi)*fitresult.b))*exp(-(u_test-fitresult.u0).^2/(2*fitresult.b^2));
    
    ind=D_fit.(T_n{n})<D_std.(T_n{n});
    y_min.(T_n{n})=D_std.(T_n{n});
    y_min.(T_n{n})(ind)=D_fit.(T_n{n})(ind)-1e-3;
    %
    %     errorbar(u_test,D_fit.(T_n{n}),y_min.(T_n{n}),D_std.(T_n{n}),'LineWidth',1)
    %     %figure;errorbar(u_test,D_fit.(T_n{n}),D_fit.(T_n{n})-D_low.(T_n{n}),D_upp.(T_n{n})-D_fit.(T_n{n}),'LineWidth',1)
    %     xlabel('Relative concentration')
    %     ylabel('D')
    %     set(gca,'Fontsize',12,'Linewidth',1)
    
    
    %Extra output plots
    
    % figure;plot(a_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[2 3 4 5]),[1 N_a]))
    % figure;plot(b_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[1 2 4 5]),[1 N_b]))
    % figure;plot(c_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[1 3 4 5]),[1 N_c]))
    % figure;plot(u0_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[1 2 3 5]),[1 N_u0]))
    % figure;plot(x0_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[1 2 3 4]),[1 N_x0]))
    
    %figure;imagesc(x0_vec.(T_n{n}),u0_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[1 2 3]),[N_u0 N_x0]))
    %figure;imagesc(c_vec.(T_n{n}),a_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[3 4 5]),[N_a N_c]))
    %figure;imagesc(b_vec.(T_n{n}),a_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[2 4 5]),[N_a N_b]))
    %figure;imagesc(u0_vec.(T_n{n}),c_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[1 3 5]),[N_c N_u0]))
    %figure;imagesc(u0_vec.(T_n{n}),a_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[2 3 5]),[N_a N_u0]))
    %figure;imagesc(c_vec.(T_n{n}),b_vec.(T_n{n}),reshape(sum(L.(T_n{n}),[1 4 5]),[N_c N_b]))
    
    %% Save data
    save('/home/mab679/output_O_HPC_norm.mat')
    
end





%
% figure
%
%
% errorbar(u_test,D_fit.l,y_min.l,D_std.l,'LineWidth',1)
% hold on
% errorbar(u_test,D_fit.m,y_min.m,D_std.m,'LineWidth',1)
% errorbar(u_test,D_fit.h,y_min.h,D_std.h,'LineWidth',1)
% set(gca,'YScale','log')
% xlabel('Relative concentration')
% ylabel('D')
% %ylim([5e-3 10])
% title(ion_text{m})
% legend('50C','100C','150C','Location','NorthWest')
% set(gca,'Fontsize',12,'Linewidth',1)







%% Plot outputs

% figure;
% errorbar(u_test,D_fit.l,D_fit.l-D_low.l,D_upp.l-D_fit.l,'LineWidth',1)
% hold on
% errorbar(u_test,D_fit.m,D_fit.m-D_low.m,D_upp.m-D_fit.m,'LineWidth',1)
% errorbar(u_test,D_fit.h,D_fit.h-D_low.h,D_upp.h-D_fit.h,'LineWidth',1)
% set(gca,'YScale','log')
% xlabel('Relative concentration')
% ylabel('D')
% ylim([5e-2 10])
% title(ion_text{m})
% legend('50C','100C','150C','Location','NorthWest')
% set(gca,'Fontsize',12,'Linewidth',1)
%
%
% figure;
% plot(u_test,D_fit.l,'b','LineWidth',1)
% hold on
% plot(u_test,D_low.l,'b--','LineWidth',1)
% plot(u_test,D_upp.l,'b--','LineWidth',1)
%
% plot(u_test,D_fit.m,'k','LineWidth',1)
% plot(u_test,D_low.m,'k--','LineWidth',1)
% plot(u_test,D_upp.m,'k--','LineWidth',1)
%
% plot(u_test,D_fit.h,'r','LineWidth',1)
% plot(u_test,D_low.h,'r--','LineWidth',1)
% plot(u_test,D_upp.h,'r--','LineWidth',1)
% ylim([5e-2 10])
%
% set(gca,'YScale','log')
% xlabel('Relative concentration')
% ylabel('D')



%feature('numcores')

