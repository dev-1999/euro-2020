import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random
### Data Cleaning
#Document containing transfermarkt market values for UEFA squads
mktval = pd.read_excel("mktval.xlsx")
mktval = mktval.set_index('Country')

#Raw copy paste data of UEFA ELO ratings
f = open("eloraw.txt", "r")
f_list = f.read().split('\n')
f.close()

#List of UEFA members
uefa = [' Albania',
' Switzerland',
' Andorra',
' Armenia',
' Austria',
' Azerbaijan',
' Belarus',
' Belgium',
' Bosnia and Herzegovina',
' Bulgaria',
' Croatia',
' Cyprus',
' Czech Republic',
' Denmark',
' England',
' Estonia',
' Faroe Islands',
' Finland',
' France',
' Georgia',
' Germany',
' Gibraltar',
' Greece',
' Hungary',
' Iceland',
' Israel',
' Italy',
' Kazakhstan',
' Kosovo',
' Latvia',
' Liechtenstein',
' Lithuania',
' Luxembourg',
' Malta',
' Moldova',
' Montenegro',
' Netherlands',
' North Macedonia',
' Northern Ireland',
' Norway',
' Poland',
' Portugal',
' Republic of Ireland',
' Romania',
' Russia',
' San Marino',
' Scotland',
' Serbia',
' Slovakia',
' Slovenia',
' Spain',
' Sweden',
' Turkey',
' Ukraine',
' Wales']
uefa = [x.strip(" ").strip('\xa0') for x in uefa]

#EURO 2020 Venues
knockout_venues = {1:'Spain',
                   8:'England',
                   5:'Hungary',
                   4:'Denmark',
                   6:'Scotland',
                   3:'England',
                   7:'Hungary',
                   2:'Netherlands',
                   17:'Germany',
                   20:'Russia',
                   19:'Italy',
                   18:'Azerbaijan',
                   25:'England',
                   26:'England',
                   29:'England'
}

#Cleaning raw ELO file into df
elo_dict = {'team':[],'elo':[],'trend':[]}
for i in range(len(f_list)):
    item = (str(f_list[i]).strip(' '))
    if item in uefa or item == 'Czechia':
        if item == 'Czechia':
            elo_dict['team'].append('Czech Republic')
        else:
            elo_dict['team'].append(f_list[i])
        #Finding ELO rating
        elo_dict['elo'].append(f_list[i+1])
        #Trend over past year (ultimately unused)
        trend = (f_list[i+5])
        if trend[0] != '+':
           # print(trend)
            trend = '-' + str(trend[-2:]).strip("’")
           # print(trend)
        elo_dict['trend'].append(trend)
elodf = pd.DataFrame.from_dict(elo_dict)
elodf['elo'] = elodf.elo.astype(int)
elodf['trend'] = elodf.trend.astype(int)
elodf = elodf.set_index('team')

#csv of all international friendly results: https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017
results = pd.read_csv('results.csv')
results['date'] =  pd.to_datetime(results['date'], format='%Y-%m-%d')
#earmarking data relevant to training SVM
results['current'] = [1 if x.year in [2019,2020,2021] else 0 for x in results.date.tolist()]
#filtering non-UEFA matches, result (for home team)
isuefa = []
outcome = []
for i in range(len(results)):
    x = 0
    y = 0
   # print(results.loc[i,'home_team'],results.loc[i,'away_team'])
    if results.loc[i,'home_team'] in uefa:
        #print(results.loc[i,'away_team'])
        if results.loc[i,'away_team'] in uefa:
            x = 1   
    isuefa.append(x)
    if results.loc[i,'home_score'] > results.loc[i,'away_score']:
        outcome.append(1)
    elif results.loc[i,'home_score'] == results.loc[i,'away_score']:
        outcome.append(0)
    else:
        outcome.append(-1)
results['isuefa'] = isuefa
results['outcome'] = outcome

#Training data, merging with elo & mv data
df = results.loc[(results.current == 1) & (results.isuefa == 1)]
df['home_elo'] = [elodf.loc[df.loc[i,'home_team'],'elo'] for i in df.index.tolist()]
df['away_elo'] = [elodf.loc[df.loc[i,'away_team'],'elo'] for i in df.index.tolist()]
df['home_mv'] = [float(str(mktval.loc[df.loc[i,'home_team'],'avgmv']).strip('$').strip('m').strip('Th.')) for i in df.index.tolist()]
df['away_mv'] = [float(str(mktval.loc[df.loc[i,'away_team'],'avgmv']).strip('$').strip('m').strip('Th.')) for i in df.index.tolist()]
df['neutral'] = [1 if x else 0 for x in df['neutral'].tolist()]


### Model Training
#generating x, y
inputs = df[['home_elo','home_mv','away_elo','away_mv','neutral']]
target = df['outcome']
#train/test split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, random_state = 0, train_size = 0.66)

model = SVC(kernel = 'linear', probability = True).fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
#accuracy = 0.6666 (spooky)


### Modeling EURO 2020 (21)
#scenarios for R16 matchups (from: https://en.wikipedia.org/wiki/UEFA_Euro_2020_knockout_phase)
matchup_matrix = pd.read_excel('matchup_matrix.xlsx',index_col = 0)
#format of results
output_table = pd.read_excel('results_template.xlsx',index_col = 0)
#group stage home teams
home_3 = ['Italy','Denmark','Netherlands','England','Spain','Germany']
home_2 = ['Russia','Scotland','Hungary']
#cleaning mktval
mktval['avgmv'] = [str(x).strip("$").strip('m').strip('Th.') for x in mktval.avgmv.tolist()]
#manual elo adjustments, format team(string):adj(int/float)
elo_adj = {
    'Netherlands':-20, #VVD
    'Belgium':-30, #KDB
    'Denmark':-50, #No striker
    'France':25 #Benzema back
}

for tm in elo_adj.keys():
    elodf.loc[tm,'elo'] += elo_adj[tm]

#home, away = string team name
#neutral = 1 if true else 0
#return [int]: -1 for home loss, 0 for draw, 1 for home win
def sim_game(home,away,neutral):
    wdl = model.predict_proba([[elodf.loc[home,'elo'], mktval.loc[home,'avgmv'],elodf.loc[away,'elo'], mktval.loc[away,'avgmv'],neutral]])[0]
    #if neutral, flip teams and average
    if neutral == 1:
        wdl2 = model.predict_proba([[elodf.loc[away,'elo'], mktval.loc[away,'avgmv'],elodf.loc[home,'elo'], mktval.loc[home,'avgmv'],neutral]])[0]
        wdl = [(wdl[i] + wdl2[2-i])/2 for i in range(3)]
    #Monte Carlo simulation according to probabilities stored in wdl
    rand = random.random()
    if rand < wdl[0]:
        return -1
    elif rand < wdl[0] + wdl[1]:
        return 0
    else:
        return 1

#group = list of four team name strings
#returns = list of points for inputted teams
def sim_group(group):
    pts = [0,0,0,0]
    for i in range(3):
        for j in range(i+1,4):
            #determining home/away/neutral
            h = -1
            a = -1
            res = 2
            if group[i] in home_3:
                h = i
                a = j
                res = sim_game(group[i],group[j],0)
            elif group[j] in home_3:
                h = j
                a = i
                res = sim_game(group[j],group[i],0)
            elif group[i] in home_2:
                h = i
                a = j
                res = sim_game(group[i],group[j],0)
            elif group[j] in home_2:
                h = j
                a = i
                res = sim_game(group[j],group[i],0)
            else:
                h = i
                a = j
                res = sim_game(group[i],group[j],1)
            #determining result and adding to point list
            if res == 1:
                pts[h] += 3
            elif res == 0:
                pts[h] += 1
                pts[a] += 1
            elif res == -1:
                pts[a] += 3
            else:
                print('ERROR! res not -1, 0, or 1')
                break
            #print('--')
    return pts         

#tmA, tmB - string of team name
#venue - number matching to appropriate venue from dictionary
#return winner, loser
def sim_knockout(tmA,tmB,venue):
    #tmA is home
    if tmA == knockout_venues[venue]:
        #simuate and advance
        outcome = sim_game(tmA,tmB,0)
        if outcome == 1:
            return tmA, tmB
        if outcome == -1:
            return tmB, tmA
        #penalty shootout - coin flip
        if outcome == 0:
            if random.random() > 0.5:
                return tmA, tmB
            else:
                return tmB, tmA
    elif tmB == knockout_venues[venue]:
        outcome = sim_game(tmB,tmA,0)
        if outcome == 1:
            return tmB, tmA
        if outcome == -1:
            return tmA, tmB
        if outcome == 0:
            if random.random() > 0.5:
                return tmA, tmB
            else:
                return tmB, tmA
    #neutral
    else:
        outcome = sim_game(tmA,tmB,1)
        if outcome == 1:
            return tmA, tmB
        if outcome == -1:
            return tmB, tmA
        if outcome == 0:
            if random.random() > 0.5:
                return tmA, tmB
            else:
                return tmB, tmA

#EURO 2021 groups
grpA = ['Turkey','Italy','Wales','Switzerland']
grpB = ['Denmark','Belgium','Finland','Russia']
grpC = ['Netherlands', 'Ukraine', 'Austria', 'North Macedonia']
grpD = ['England', 'Croatia', 'Scotland', 'Czech Republic']
grpE = ['Spain', 'Sweden', 'Poland', 'Slovakia']
grpF = ['Hungary','Portugal','France','Germany']

### MAIN LOOP
#number of iterations
n = 2500
for iteration in range(n):
    #sim group
    winners = []
    runnerups = []
    thirds = []
    thirdpts = []
    fourths = []
    for grp in [grpA,grpB,grpC,grpD,grpE,grpF]:
        points = sim_group(grp)
        grp_sorted = [x for y, x in sorted(zip(points, grp))]
        pts_sorted = [y for y, x in sorted(zip(points, grp))]
        winners.append(grp_sorted[-1])
        runnerups.append(grp_sorted[-2])
        thirds.append(grp_sorted[-3])
        thirdpts.append(pts_sorted[-3])
        fourths.append(grp_sorted[0])
    thirdelo = [elodf.loc[x,'elo'] for x in thirds]
    thirdtb = [thirdpts[i] + thirdelo[i]/10000 for i in range(6)]
    third_sorted = [x for y, x in sorted(zip(thirdtb, thirds))]
    third_grps = [x for y, x in sorted(zip(thirdtb, ['A','B','C','D','E','F']))]
    scenario = ""
    for x in sorted(third_grps[-4:]):
        scenario = scenario + x
    scenario
    third_dict = dict(zip(third_grps,thirds))
    group_losers = fourths + third_sorted[:2]


    # In[733]:


    #setup r16
    tm1 = winners[1]
    tm16 = third_dict[matchup_matrix.loc[scenario,'B'].strip('3')]
    tm8 = winners[0]
    tm9 = runnerups[2]
    tm5 = winners[5]
    tm12 = third_dict[matchup_matrix.loc[scenario,'F'].strip('3')]
    tm4 = runnerups[3]
    tm13 = runnerups[4]
    tm6 = winners[4]
    tm11 = third_dict[matchup_matrix.loc[scenario,'E'].strip('3')]
    tm3 = winners[3]
    tm14 = runnerups[5]
    tm7 = winners[2]
    tm10 = third_dict[matchup_matrix.loc[scenario,'C'].strip('3')]
    tm2 = runnerups[0]
    tm15 = runnerups[1]

    #sim r16, setup QF
    r16_losers = []
    tm17, l16 = sim_knockout(tm1,tm16,1)
    r16_losers.append(l16)
    tm24, l16 = sim_knockout(tm8,tm9,8)
    r16_losers.append(l16)
    tm20, l16 = sim_knockout(tm5,tm12,5)
    r16_losers.append(l16)
    tm21, l16 = sim_knockout(tm4,tm13,4)
    r16_losers.append(l16)
    tm19, l16 = sim_knockout(tm6,tm11,6)
    r16_losers.append(l16)
    tm22, l16 = sim_knockout(tm3,tm14,3)
    r16_losers.append(l16)
    tm18, l16 = sim_knockout(tm7,tm10,7)
    r16_losers.append(l16)
    tm23, l16 = sim_knockout(tm2,tm15,2)
    r16_losers.append(l16)

    #sim qf, setup SF
    r8_losers = []
    tm25, l8 = sim_knockout(tm17,tm24,17)
    r8_losers.append(l8)
    tm28, l8 = sim_knockout(tm20,tm21,20)
    r8_losers.append(l8)
    tm26, l8 = sim_knockout(tm19,tm22,19)
    r8_losers.append(l8)
    tm27, l8 = sim_knockout(tm18,tm23,18)
    r8_losers.append(l8)

    #sim semis, final
    sf_losers = []
    tm29, lsf = sim_knockout(tm25,tm28,25)
    sf_losers.append(lsf)
    tm30, lsf = sim_knockout(tm26,tm27,26)
    sf_losers.append(lsf)
    winner, runnerup = sim_knockout(tm29,tm30,29)

    #keep running total of results
    for i in group_losers:
        output_table.loc[i,'P_GROUP'] += 1/n
    for i in r16_losers:
        output_table.loc[i,'P_R16'] += 1/n
    for i in r8_losers:
        output_table.loc[i,'P_QF'] += 1/n
    for i in sf_losers:
        output_table.loc[i,'P_SF'] += 1/n
    output_table.loc[runnerup,'P_LF'] += 1/n
    output_table.loc[winner,'P_WF'] += 1/n
    print(iteration,winner,runnerup)


# In[740]:


#output
output_table.to_excel('sim_results.xlsx')


# In[ ]:




