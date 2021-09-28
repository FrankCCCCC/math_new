%{
P491 10.
f(x) = 
    4 - x, if 0 < x < 4
    -4 - x, if -4 < x < 0
for -4 < x < 4, p=4
%}
fn1 = @(x) 8/pi * (sin(pi*x/4) + (1/2) * sin(pi*x/2) + (1/3) * sin(3*pi*x/4) + (1/4) * sin(pi*x) + (1/5) * sin(5*pi*x/4));
f1 = figure;
figure(f1);
fplot(fn1, [-pi, pi])
title('Problem 10')
ylabel('f(x)')
xlabel('x')
saveas(gcf,'plot_problem10.png')

%{
P491 11.
f(x) = x^2, -1 < x < 1, p=2
%}
fn2 = @(x) (1/3) + (4/(pi^2)) * (-cos(pi*x) + (1/4) * cos(2*pi*x) - (1/9) * cos(3*pi*x) + (1/16) * cos(4*pi*x) - (1/25) * cos(5*pi*x));
f2 = figure;
figure(f2);
fplot(fn2, [-1, 1])
title('Problem 11')
ylabel('f(x)')
xlabel('x')
saveas(gcf,'plot_problem11.png')

%{
P491 24.
f(x) = 
    1, if 2 < x < 4
    0, if 0 < x < 2
for -4 < x < 4, p=4 (Half Wave)
%}
fn24_odd = @(x) (2/pi) * (sin(pi*x/4) - sin(pi*x/2) + (1/3) * sin(3*pi*x/4) + (1/5) * sin(5*pi*x/4) - (1/3) * sin(3*pi*x/2));
f24_odd = figure;
figure(f24_odd);
fplot(fn24_odd, [-4, 4])
title('Problem 24 - Odd Extension')
ylabel('f(x)')
xlabel('x')
saveas(gcf,'plot_problem24_odd.png')

fn24_even = @(x) 1/2 + (2/pi) * (-cos(pi*x/4) + (1/3) * cos(3*pi*x/4) - (1/5) * cos(5*pi*x/4) + (1/7) * cos(7*pi*x/4) - (1/9) * cos(9*pi*x/4));
f24_even = figure;
figure(f24_even);
fplot(fn24_even, [-4, 4])
title('Problem 24 - Even Extension')
ylabel('f(x)')
xlabel('x')
saveas(gcf,'plot_problem24_even.png')

%{
P491 28.
f(x) = x
for 0 <= x <= L, p=2L (Half Wave)
%}
fn28_odd = @(x) (2/pi) * (sin(pi*x) - (1/2) * sin(2*pi*x) + (1/3) * sin(3*pi*x) - (1/4) * sin(4*pi*x) + (1/5) * sin(5*pi*x));
f28_odd = figure;
figure(f28_odd);
fplot(fn28_odd, [-1, 1])
title('Problem 28 - Odd Extension')
ylabel('f(x)')
xlabel('times of L')
saveas(gcf,'plot_problem28_odd.png')

fn28_even = @(x) 1/2 - (4/(pi * pi)) * (cos(pi*x) + (1/9) * cos(3*pi*x) - (1/25) * cos(5*pi*x) + (1/49) * cos(7*pi*x) - (1/81) * cos(9*pi*x));
f28_even = figure;
figure(f28_even);
fplot(fn28_even, [-1, 1])
title('Problem 28 - Even Extension')
ylabel('f(x)')
xlabel('times of L')
saveas(gcf,'plot_problem28_even.png')

%{
Problem III
%}
fnIII = @(x) cos(2 * x) + 3 * sin(2 * x);
fIII = figure;
figure(fIII);
fplot(fnIII, [-pi, pi])
title('Problem III')
ylabel('cos(2t) + 3 * sin(2t)')
xlabel('t')
saveas(gcf,'plot_problemIII_fs.png')


fnIII2 = @(x) (1/(cos(atan(-3)))) * cos(2 * x + atan(-3));
fIII2 = figure;
figure(fIII2);
fplot(fnIII2, [-pi, pi])
title('Problem III')
ylabel('X * cos(2t + Y), X=1/cos(arctan(-3)), Y=arctan(-3)')
xlabel('t')
saveas(gcf,'plot_problemIII_fn.png')
