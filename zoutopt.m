tic
zoutopt_1(@funz)
toc

function [xopt,fopt,iter] = zoutopt_1(funz) 

% Initialization parameters
x0 = [0.5 0.5]; 
xl = [0 0]; 
xu = [5 5]; 

% Based on Input functions & constraints calculating the gradient functions
delf = @(x) [2*x(1) - 6; 2*x(2) - 8]; 
delg = @(x) [3; 5];

% number of constraints
nc = 0; ncs = 1; 

crit = 1e-4; 
kmax = 1e3;
mcrit = 2e-3; 
gcrit = 1e-4; 
maxs = 30;
nv = length(x0); 
x = x0; 
fg = funz(x); 
f = fg(1); g = fg(2); ic = 0; fold = f; iter = 0; f0 = f;

T = table;
while (1)
    iter = iter + 1;
    [nc, na] = actcont(ncs, crit, g);
    if (iter > kmax) 
      disp('Maximum possible iterations exceeded.');
      break; 
    end
    [d, dn, beta] = dirvec(delf,delg,crit,nv,x,nc,xl,xu,mcrit); 
    if (abs(dn) < crit || abs(beta) < crit), break; end
    d = d/dn;
    alpha = linsr(funz,delf,nv,ncs,x,nc,na,xl,xu,d,x0,maxs,gcrit,crit); %
      
    x = x0 + alpha*d; 
    fg = funz(x); 
    f = fg(1); 
    x0 = x;
    if (abs(f - fold) < crit)
        ic = ic + 1;
        if (ic == 2) 
            break; 
        end
    else
        ic = 0;
    end
    fold = f;
    x_t = table(iter, x(1), x(2), f);
    T = [T; x_t];
end
    fg = funz(x); 
    f = fg(1); 
    g = fg(2); 
    [nc,na] = actcont(ncs, crit, g); 
    xopt = x; 
    fopt = f;
    
    T.Properties.VariableNames = {'Iterations' 'x_1' 'x_2' 'funcValue'};
    disp(T);
end


function [nc, na] = actcont(ncs, mcrit, g)
    for i = 1:ncs 
        na(i) = i; 
    end
    nc = 0;
    for k = 1:ncs
        if ( g(k) > -mcrit), nc = nc + 1; ntemp = na(k); na(k) = na(nc);
          na(nc) = ntemp; 
        end
    end
end

function [d, dn, beta] = dirvec(delf,delg,crit,nv,x,nc,xl,xu,mcrit)
df = delf(x)'; % row vector
if (nc > 0), A = delg(x)'; end
df = df/norm(df); % normalization
if (nc > 0)
    for j = 1:nc, A(j,:) = A(j,:)/sqrt(A(j,:)*A(j,:)'); end
end
% active bounds
for k = 1:nv
if (xl(k) - x(k) + mcrit >= 0), nc = nc + 1; A(nc,1:nv) = 0; A(nc,k) = -1; end
end
for k = 1:nv
if (x(k) - xu(k) + mcrit >= 0), nc = nc + 1; A(nc,1:nv)=0; A(nc,k) = 1; end
end
if (nc == 0), beta = 1; d = -df; dn = norm(d); return; end
[beta, d] = simpx(nc,nv,df,A); dn = sqrt(d*d'); % feasible direction
end


function alpha = linsr(funz,delf,nv,ncs,x,nc,na,xl,xu,d,x0,maxs,gcrit,crit) 
nlarge = 1e40; c = max(abs(xu - xl));
for k = 1:nv
    if (abs(d(k))*nlarge > c)
        if (d(k) < 0)
            cn = (xl(k) - x(k)) / d(k);
            if (cn < nlarge), nlarge = cn; end
        else
            cn = (xu(k) - x(k)) / d(k);
            if (cn < nlarge), nlarge = cn; end
        end
    end
end
abet = nlarge; x = x0 + abet * d; fg = funz(x); f = fg(1); g = fg(2);
  gmax = max(g); inda = 1;
if (gmax <= 0), inda = 0; end
if (inda == 0), amax = abet;
else, x1 = 0; [xm, fm] = nears(funz,x1,abet,x,d,x0,maxs,crit); amax = xm;
end
x = x0 + amax*d; df = delf(x)'; sdr = df*d';
if (sdr <= 0), alpha = amax; return; end
a1 = 0; a2 = amax; adif = a2 - a1;
while ((a2 - a1) > crit * adif)
    am = (a1 + a2)/2; x = x0 + am*d; df = delf(x)'; sdr = df*d';
    if (sdr == 0), break; end
    if (sdr < 0), a1 = am;
    elseif (sdr > 0), a2 = am;
    end
end
alpha = a1;
end


function [xm, fm] = nears(funz,xa,xb,x,d,x0,maxs,crit)
miter = 0;
fa=0;
while (1)
    xm = (xa + xb)/2; 
    miter = miter + 1;
    if (miter > maxs) 
        xm = xa; 
        fm = fa; 
        return; 
    end
    x = x0 + xm*d; 
    fg = funz(x); 
    f = fg(1); 
    g = fg(2); 
    gmax = max(g); 
    fm = f; 
    if (gmax <= 0 && gmax >= -crit) 
        return;
    end
    if (gmax < 0) 
        xa = xm; 
        fa = fm;
    else
        xb = xm; 
    end
end
end


function [beta, d] = simpx(nc,nv,df,A)
    % Find search direction using the linear programming (simplex) method
    Bg = 1e2; nrow = nc + nv + 2; nm = nc + nv + 1; Bm(1:nrow) = 0; Bm(1) = sum(df);
    for j = 1:nc
        Bm(j+1) = 0;
        for k = 1: nv 
            Bm(j+1) = Bm(j+1) + A(j,k); 
        end
    end
    for k = nc + 2:nrow - 1 
        Bm(k) = 2; 
    end
    for k = 1: nm 
        Bs(k) = nv + k + 1; 
    end
    ncol = nv + nm + 1;
    for k = 1:nm
        if (Bm(k) < 0)
            ncol = ncol + 1; 
            Bs(k) = -ncol; 
        end
    end
    
    Am(1:nrow,1:ncol) = 0; 
    Am(1,1:nv) = df; 
    Am(1,nv+1) = 1;
    for k = 1:nc
        for j = 1: nv 
            Am(k+1,j) = A(k,j); 
        end
        Am(k+1,nv+1) = 1;
    end
    
    mi = 0;
    for k = nc+2: nrow-1 
        mi = mi + 1; 
        Am(k,mi) = 1; 
    end
    
    Am(nrow,nv+1) = -1;
    for k = 1: nm 
        Am(k,nv+k+1) = 1; 
    end
    nt = nv + nm + 1;
    for k = 1: nm
        if (Bm(k) < 0)
            nt = nt + 1; Bm(k) = -Bm(k);
            for j = 1:ncol
                Am(k,j) = -Am(k,j); 
            end
            Am(k,nt) = 1; Am(nrow,nt) = Bg;
        end
    end

for k = 1:nm
    if (Bs(k) < 0)
        Bs(k) = -Bs(k);
        for j = 1: ncol 
            Am(nrow,j) = Am(nrow,j) - Bg*Am(k,j); 
        end
        Bm(nrow) = Bm(nrow) - Bg*Bm(k);
    end
end

while (1)
    nf0 = 0;
    for k = 1:ncol
        if (Am(nrow, k) < 0) 
            nf0 = 1; 
            break; 
        end
    end
    if (nf0 == 0)
        break; 
    end
    c = Bg;
    for j = 1:ncol
        if (Am(nrow, j) < c)
            c = Am(nrow, j); 
            iv = j; 
        end
    end
    ik = 0; jk = 0;
    for k = 1:nrow - 1
        if (Am(k,iv) > 0)
            jk = jk + 1; c1 = Bm(k)/(Am(k,iv) + 1e-10);
            if (jk == 1), c = c1; jp = k;
            else, if (c1 < c), c = c1; jp = k; end
            end
            ik = 1;
        end
    end
    Bs(jp) = iv;
    if (ik == 0), disp('Unbounded objective function.'); break; end
    c1 = 1/Am(jp,iv); Bm(jp) = c1*Bm(jp);
    for j = 1:ncol, Am(jp,j) = c1*Am(jp,j); end
    for k = 1: nrow
        if (k ~= jp)
            c2 = Am(k,iv);
            for j = 1: ncol, Am(k,j) = Am(k,j) - c2*Am(jp,j); end
            Bm(k) = Bm(k) - c2*Bm(jp);
        end
    end
end

d(1:nv) = -1;
for k = 1: nm
    for j = 1: nv
        if (j == Bs(k)), d(j) = Bm(k) - 1; end
    end
end
beta = Bm(nrow);
end