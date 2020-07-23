function [u_out,v_out,x]=Diffusion_numeric_1D_mid(x_in,D_gen,flip,delta_t,L,N_x,T)
%Model for 1D diffusion

%Based on http://www.atmos.albany.edu/facstaff/brose/classes/ATM623_Spring2015/Notes/Lectures/Lecture16%20--%20Numerical%20methods%20for%20diffusion%20models.html


%Old setup commented out and instead fed in as parameters to the function
%D_gen= @(u_D) a+c*exp(-(u_D-u0).^2/(2*b^2));       %Gaussian
%D_gen= @(u_D) (u_D>u1 & u_D<u2)*c +a;              %Step layer
%D_gen= @(u_D) (u_D>u1 & u_D<u2)*c +a+(u_D>=u2)*c_sig;  %Three layer
%D_gen= @(u_D) (u_D>u0)*a +c;                        %Single step
%D_gen= @(u_D) c_sig./(1+exp(-(u_D-u0)/b_sig))+a;           %Sigmoid
%D_gen= @(u_D) c_sig./(1+exp(-(u_D-u0)/b_sig))+a+c*exp(-(u_D-u0).^2/(2*b^2));
% N_x=301;
% L=150;
% delta_t=5e-3;
% T=300;
% %N_t=11;
% %D=0.01;
% u_test=[0:1e-3:1];
% %a0=0.96;
% %b0=0.05;
% %c0=0.65;
% %u0=0.46;
% D_test=D_gen(u_test);

%N_plot=50;


%Initialise variables
N_t=floor(T/(delta_t)+1);
delta_x=L/(N_x);
x_stag=linspace(-L/2,L/2,N_x+1);
x=x_stag(1:end-1)+delta_x/2;



% Check the solution does not diverge
% beta=max(D_test)*delta_t/delta_x^2; %Less than 0.5 to be stable
%
% if beta>0.5
%     warning('Too small spatial step/ too big time step')
% end
%
% figure;plot(u_test,D_test)


%Initial conditions
u=zeros(size(x));
if flip==1
    u(floor(N_x/2)+1:end)=1;
elseif flip==0
    u(1:floor(N_x/2))=1;
else
    error('Incorrect b.c. setup with flip variable')
end



% figure
%
% h=plot(x,u);
% hold on
% plot([0 0],[0 1],'k--')
% counter_max=floor(N_t/N_plot);
% counter=1;

%Check total number is conserved
%u_sum=NaN*zeros(N_t,1);
%u_sum(1)=sum(u);

v_out=NaN*zeros(N_x,N_t);
v_out(:,1)=u;

for n=1:N_t
    
    dudx=diff(u)./delta_x;
    
    %No flux at the ends boundary condition
    %dudx(1)=0;
    %dudx(N_x)=0;
    dudx=[0,dudx,0];
    
    %Calculate diffusivity at each point    
    u_D=[u,u(end)];
    u_D2=[u(1),u];
    %a0=0.96;
    %b0=0.05;
    %c0=0.65;
    %x0=0.46;
    %D=a+c*b./((u_D-u0).^2+b^2);
    %D=a+c*normpdf(u_D,u0,b);
    D=D_gen((u_D+u_D2)/2); %Nonsense at the boundaries? but works elsewhere
    
    %Calculate flux (kind of)
    F=-D.*dudx;
    
    %Calculate the time derivative
    dudt=diff(F)./(-delta_x);
    
    %Debugging plot
    %figure;plot(x,u)
    %hold on
    %plot(x,dudt)
    
    %Move to next time step
    u=u+delta_t*dudt;
    
    %Plot the result
    %         if counter<counter_max
    %             counter=counter+1;
    %         else
    %             h.YData=u;
    %             drawnow
    %             counter=1;
    %         end
    %
    %u_sum(n+1)=sum(u);
    
    v_out(:,n)=u;
end
% Update the plot
% h.YData=u;
% drawnow

%Interpolate the data onto the provided x points
u_out=interp1(x,u,x_in,'linear',0);

%Handle points outside of range to stop errors
inds=x_in<min(x);
if (flip==1)
    u_out(inds)=0;
else
    u_out(inds)=1;
end

inds=x_in>max(x);
if (flip==1)
    u_out(inds)=1;
else
    u_out(inds)=0;
end

end