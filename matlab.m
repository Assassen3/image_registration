A = imread("data\tomato\20230315_p1_2_depth.png");

A = double(A)

A = A / 1.0 / max(max(A));

imshow(A)