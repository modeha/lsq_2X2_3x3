clear all
clc
%delete(strcat('/Users/Mohsen/Documents/nlpy_mohsen/lsq/matlab','.txt'));
    dim=[ 1           2           1
          4           3           2
          8           6           4
          16          12          10
          24          18          12
          32          14          13
          32          22          16
          32          24          23
          40          30          20
          48          36          24
          56          42          28
          64           8           6
          64          48          32
          72          54          36
          80          60          40
          88          66          44
          96          72          48
         112          84          56
         128          16          12
         304          88          60
         696         512         400
        1024         500         450
        1024        1500        1470
        1092        1000         800
        1268         996         400
        1280        1600         400
        1300        1000         900
        1392        1024         800
        1944        1568         600
        ];
    dim = sortrows(dim);
         %dim = [100 30];
%     dim = [4096,2048
%         ];
    n = size(dim)
for i = 1:n(1)

% a=[dim(i,1),dim(i,2)];
% a

    simple(rand(10),'type','direct','m',dim(i,1),'n',dim(i,2))
    simple(rand(10),'type','sparse','n',dim(i,1),'m',dim(i,2),'sparse',dim(i,3))
    simple(rand(10),'type','gauss','n',dim(i,1),'m',dim(i,2))
    %%simple(rand(10),'testproblems',1,'k',dim(i,2))
    %%simple(rand(10),'Sparsity','Ture','Solver_Type', 'InDirect','type','read_from_file','n',dim(i,1),'m',dim(i,2))
end

dim2=[8,6,4;
    32,4,3;
    64,8,6;
    16,12,10;
    128,12,12;
    256,32,32;
    304,88,70;
    512,64,55;
    536,192,180;
    1024,128,120;
    2048,256,240;
    4096,512,500;
    1072,384,350;
    1576,1117,990;
    2144,1768,1600;
    2288,1536,1450;
    3192,2024,1900;
    3768,3096,1900];