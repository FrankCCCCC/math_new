%{
P482 6.
f(x) = |x|
%}
fn1 = @abs;
f1 = figure;
figure(f1);
fplot(fn1, [-pi, pi])
saveas(gcf,'abs.png')

%{
P482 14.
f(x) = x^2
%}
fn2 = @(x) pi.^2/3 - 4 * cos(x) + cos(2 * x) - 4 / 9 * cos(3 *x) + 1 / 4 * cos(4 * x) - 4 / 25 * cos(5 * x);
f2 = figure;
figure(f2);
fplot(fn2, [-pi, pi])
saveas(gcf,'square.png')