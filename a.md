flowchart TD
    A([Старт]) --> B[Задано: f(x), интервал [a,b] \n с f(a)*f(b) < 0, tol]
    B --> C{ |f(a)| < |f(b)| ? }
    C -- да --> C1[Поменять a<->b, fa<->fb] --> D
    C -- нет --> D[ c = a; fc = f(a);\n mflag = true; d = b-a ]
    D --> E{ |b-a| > tol \n и |f(b)| > tol ? }
    E -- нет --> Z([Стоп: x≈b ])
    E -- да --> F{ fa ≠ fc \n и fb ≠ fc ? }
    F -- да --> G[ s = IQI(a,b,c,fa,fb,fc) ]
    F -- нет --> H[ s = b - fb*(b-a)/(fb-fa) \n  (секущая) ]
    G --> I
    H --> I{ Проверка «безопасности» шага:\n1) s ∉ ((3a+b)/4 , b)\n2) mflag и |s-b| ≥ |b-c|/2\n3) !mflag и |s-b| ≥ |c-d|/2\n4) |b-c| < tol или |c-d| < tol }
    I -- да --> J[ s = (a+b)/2 \n (бисекция); mflag = true ]
    I -- нет --> K[ mflag = false ]
    J --> L
    K --> L
    L[ fs = f(s); d = c; c = b; fc = fb ] --> M{ fa*fs < 0 ? }
    M -- да --> N[ b = s; fb = fs ]
    M -- нет --> O[ a = s; fa = fs ]
    N --> P{ |fa| < |fb| ? }
    O --> P{ |fa| < |fb| ? }
    P -- да --> Q[ Поменять a<->b, fa<->fb ]
    P -- нет --> R[ (оставить как есть) ]
    Q --> E
    R --> E
