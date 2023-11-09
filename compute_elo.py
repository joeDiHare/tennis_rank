import requests
import pandas as pd
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import sys
from loguru import logger


URL_BASE = "https://app.tennisrungs.com"
    

class ComputeElo:
	'''
	Class to implement the Elo rating. Formulas & parameters are taken from:
	https://en.wikipedia.org/wiki/Elo_rating_system
	'''
	def __init__(self, k=20, g=5):
		self.elo = {}
		self.elo_history = []
		self.k = k  # this should be >20 when we have small number of matches
		self.g = g  # dunno about this

	def add_players(self, names, rating=1500):
		for name in names: self.elo[name] = rating

	def expected_outcome(self, p1, p2, a=400.0, b=10.0):
		return 1 / (b**((p2 - p1) / a) + 1)

	def log_result(self, winner, loser):
		exp_result = self.expected_outcome(self.elo[winner], self.elo[loser])
		self.elo[winner] = max(100, self.elo[winner] + (self.k * self.g) * (1 - exp_result))
		self.elo[loser]  = max(100, self.elo[loser]  + (self.k * self.g) * (exp_result - 1))
		self.elo_history.append(list(self.elo.values()))

	def rankings(self):
		return sorted(self.elo.items(), key=lambda item: -item[1])

	def top(self, N=None, verbose=False):
		rankings = self.rankings()[:N or len(self.elo)]
		if verbose:
			for n, (a, b) in enumerate(rankings, 1):
				print(f"#{n} {b:.0f} {a}")
		return rankings

	def generate_history(self):
		df = pd.DataFrame(self.elo_history)
		df.columns = self.elo.keys()
		return df

def parse_table(table):
    '''Parse table from scraped html.'''
    data = []
    for row in table.find_all('tr'):
        row_data = []
        for cell in row.find_all('td'):
            row_data.append(cell.text.strip())
            if cell.a is not None:
                row_data.append(cell.a['href'])
        data.append(row_data)
    return data[1:]

def calc_ratio_win(s, add_tb=False):
    '''Needs to be improved. Extract winning stats.'''
    if 'Forfeit' in s:
        # Ignore forfeit
        return 0.
    groups = [g.strip() for g in s.split(',')]
    num, den = 0, 0
    for g in groups[:2]:
        a, b = g[:3].split('-')
        num += int(a)
        den += int(a) + int(b)
    # if add_tb and len(groups)==3:
    #     if groups[2][0]=='(':
    #         g, d = groups[2][1:-1], 4
    #     else:
    #         g, d = groups[2], 1
    #     a, b = g.split('-')
    #     # Roughly equate 1 TB won to 2.5 games
    #     num += 2.5
    #     den += 2.5
    return num/den if den>0 else 0


def cleanup_result_logs(df, df_player):
    df.columns = ['player_a_id', 'player_a',  'date', 'player_b', 'link', 'res', 'score']
    # Add players that dropped fromt the challenge:
    df['player_b_id'] = df['link'].apply(lambda x: x.split('teamId=')[-1])
    for _, row in df.iterrows():
        if row['player_b_id'] not in df_player['player_id'].values:
            logger.debug(f'Adding dropped-player "{row["player_b"]}" to list.')
            df_player.loc[len(df_player.index)] = [row['player_b'], row['link'], None, row['player_b_id']] 
    df['player_b'] = df['player_b_id'].apply(lambda x: df_player[df_player['player_id']==x].iloc[0]['name'])
    # Add useful fields:
    df['winner'] = df['res'].apply(lambda x: 'a' if x=='W' else 'b')
    df['loser'] = df['res'].apply(lambda x: 'b' if x=='W' else 'a')
    df['winner_name'] = df.apply(lambda x: x[f'player_{x["winner"]}'], axis=1)
    # Create winning stats, maybe useful in the future:
    df['ratio_win'] = df['score'].apply(lambda x: calc_ratio_win(x))
    df['ratio_win'] = df.apply(lambda x: x['ratio_win'] if x['res']=='W' else 1-x['ratio_win'], axis=1)
    df['ratio_win_tb'] = df['score'].apply(lambda x: calc_ratio_win(x, add_tb=True))
    df['ratio_win_tb'] = df.apply(lambda x: x['ratio_win_tb'] if x['res']=='W' else 1-x['ratio_win_tb'], axis=1)
    # Unique identidifer for a match:
    df['match_uid'] = df.apply(lambda d: f"{d['date'].replace('/', '')}_{'_'.join(sorted(d[['player_a_id', 'player_b_id']]))}", axis=1)
    # Sort chronologically:
    df['date'] = pd.to_datetime(df['date'])
    df['ts'] = df['date'].apply(lambda x: pd.Timestamp(x).timestamp())
    df['week_nr'] = df['date'].dt.isocalendar().week
    df = df.sort_values('ts').reset_index(drop=True)
    df = df[['date', 'player_a', 'player_a_id', 'player_b', 'player_b_id', 'winner', 'loser', 'score',
            'ratio_win', 'ratio_win_tb', 'match_uid', 'winner_name', 'ts', 'week_nr']]
    logger.debug(f'{len(df)} total records (counting duplicates)')
    # De-duplicate logs:
    df = df[~df.duplicated(subset=['match_uid'], keep='first')].reset_index(drop=True)
    logger.debug(f'{len(df)} after de-duplication')
    return df


def crawl_players_page(args):
    url = f"{URL_BASE}/atc/tennis-ladders/{args.league_id}"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    table = soup.find('table', {'id': 'ladderrankings'})
    data = parse_table(table)
    df_player = pd.DataFrame([(x[1], x[2], x[6]) for x in data])
    df_player.columns = ['name', 'link', 'points']
    df_player['player_id'] = df_player['link'].apply(lambda x: x.split('teamId=')[-1])
    return df_player


def crawl_results_pages(df_player):
    results = []
    for idx, row in df_player.iterrows():
        req = requests.get(f"{URL_BASE}{row['link']}")
        soup = BeautifulSoup(req.text, "html.parser")
        table = soup.find('table', {'id': 'challenges'})
        if table is not None:
            res_ = parse_table(table)
            res_ = [[row['player_id'], row['name']] + r for r in res_]
            results.append(res_)
    df = pd.DataFrame([y for x in results for y in x])
    return df


def main(args):
    """ Main entry point of the app """
    logger.info(f"Crawling data for {args.league_id} and computing ELO score for all players.")
    
    # Crawl players
    logger.info("[1/3] Crawling list of players ...")
    df_player = crawl_players_page(args)

    # Create history by crawling all pages and logging results
    logger.info("[2/3] Creating log with all matches played ...")
    df_results = crawl_results_pages(df_player)
    df_results = cleanup_result_logs(df_results, df_player)
    logger.info(f'Found {len(df_player)} players with {len(df_results)} matched played.')

    # Compute ELO score
    logger.info("[3/3] Calculate ELO score for all players ...\n")
    CE = ComputeElo()
    CE.add_players(df_player['name'], rating=1500)
    # Iterate through all records and update elo's
    for idx, row in df_results.sort_values('ts').iterrows():
        CE.log_result(winner=row[f'player_{row["winner"]}'],
                      loser=row[f'player_{row["loser"]}'])

    # Print ranks
    rankings = CE.top(verbose=False)
    logger.info(f'Listing all {len(df_player)} players and their ELO score:')
    for n, (a, b) in enumerate(rankings, 1):
        logger.info(f"#{n} {b:.0f} {a}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--league_id", default="singles-3.0-3.5/143889747",
                        help="Required positional argument")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, colorize=True,
               format='<green>{time:HH:mm:ss}</green> | <level>{level: <8} | {message}</level>',
               level="DEBUG" if args.verbose else "INFO")
    main(args)