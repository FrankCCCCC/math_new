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
saveas(gcf,'plot_subject10.png')

%{
P491 11.
f(x) = x^2, -1 < x < 1, p=2
%}
fn2 = @(x) (1/3) + (4/(pi^2)) * (-cos(pi*x) + (1/4) * cos(2*pi*x) - (1/9) * cos(3*pi*x) + (1/16) * cos(4*pi*x) - (1/25) * cos(5*pi*x));
f2 = figure;
figure(f2);
fplot(fn2, [-1, 1])
saveas(gcf,'plot_subject11.png')

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
saveas(gcf,'plot_subject24_odd.png')

fn24_even = @(x) 1/2 + (2/pi) * (-cos(pi*x/4) + (1/3) * cos(3*pi*x/4) - (1/5) * cos(5*pi*x/4) + (1/7) * cos(7*pi*x/4) - (1/9) * cos(9*pi*x/4));
f24_even = figure;
figure(f24_even);
fplot(fn24_even, [-4, 4])
saveas(gcf,'plot_subject24_even.png')

