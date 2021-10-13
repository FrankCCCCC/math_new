%{
P91 3.
y_p = 1.0625 cos 2t + 3.1875 sin 2t
%}
fn1 = @(x) 1.0625 * cos(2 * x) + 3.1875 * sin(2 * x);
f1 = figure;
figure(f1);
fplot(fn1, [0, 3*pi])
title('Problem 3')
ylabel('y_p = 1.0625 cos 2t + 3.1875 sin 2t')
xlabel('t')
saveas(gcf,'plot_problem3.png')

%{
P91 10.
y = A cos 4t + B sin 4t + 7t sin 4t
%}
A = 1;
B = 1;
fn2 = @(x) A * cos(4 * x) + B * sin(4 * x) + 7 * x * sin(4 * x);
f2 = figure;
figure(f2);
fplot(fn2, [0, 3*pi])
title('Problem 10')
ylabel('y = A cos 4t + B sin 4t + 7t sin 4t, A=B=1')
xlabel('t')
saveas(gcf,'plot_problem10.png')
