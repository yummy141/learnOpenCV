vector<pairs*> MatchPairs;
ifstream fin;
fin.open("matchpairs.in");
for (int n = 0; n < 74; n++){
pairs* tp = new pairs;
double t;
for (int i = 0; i < 3; i++){
    for (int j = 0; j < 4; j++){
        fin >> t;
        if(j < 3)
            tp->R(i, j) = t;
        else
            tp->t(i, 0) = t;
    }
}
cout << tp->R << endl;
cout << tp->t << endl;
MatchPairs.emplace_back(tp);
}
fin.close();
