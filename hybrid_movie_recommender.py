"""
Hybrid Movie Recommendation System
===================================
This system combines Content-Based Filtering and Collaborative Filtering
to provide personalized movie recommendations.

Author: Hybrid Recommender Team
Date: 2026
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: DATA LOADING AND GENERATION
# =============================================================================

def load_or_generate_data():
    """
    Load MovieLens dataset or generate sample data if not available.
    Returns movies DataFrame and ratings DataFrame.
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    try:
        # Create comprehensive dataset with 150 movies (including Indian regional films)
        movies_list = [
            # ==========================================
            # BOLLYWOOD (HINDI) MOVIES - ID 1-30
            # ==========================================
            {'movieId': 1, 'title': 'Dilwale Dulhania Le Jayenge', 'genres': 'Drama|Romance', 'keywords': 'bollywood love train parents', 'tags': 'hindi classic romance'},
            {'movieId': 2, 'title': '3 Idiots', 'genres': 'Comedy|Drama', 'keywords': 'college engineering friendship', 'tags': 'hindi comedy inspirational'},
            {'movieId': 3, 'title': 'Lagaan', 'genres': 'Adventure|Drama|Musical', 'keywords': 'cricket british india', 'tags': 'hindi historical sports'},
            {'movieId': 4, 'title': 'Dangal', 'genres': 'Action|Biography|Drama', 'keywords': 'wrestling daughters father', 'tags': 'hindi biographical sports'},
            {'movieId': 5, 'title': 'Sholay', 'genres': 'Action|Adventure|Crime', 'keywords': 'criminals bandits revenge', 'tags': 'hindi classic action'},
            {'movieId': 6, 'title': 'Mughal-e-Azam', 'genres': 'Drama|Historical|Musical', 'keywords': 'emperor courtesan love', 'tags': 'hindi classic historical'},
            {'movieId': 7, 'title': 'Gabbar Is Back', 'genres': 'Action|Crime|Drama', 'keywords': 'vigilante revenge crime', 'tags': 'hindi action thriller'},
            {'movieId': 8, 'title': 'PK', 'genres': 'Comedy|Drama|Sci-Fi', 'keywords': 'alien india religion', 'tags': 'hindi comedy satire'},
            {'movieId': 9, 'title': 'Bajrangi Bhaijaan', 'genres': 'Action|Comedy|Drama', 'keywords': 'pakistan child lost', 'tags': 'hindi emotional action'},
            {'movieId': 10, 'title': 'My Name is Khan', 'genres': 'Drama|Romance', 'keywords': 'autism terrorism love', 'tags': 'hindi drama emotional'},
            {'movieId': 11, 'title': 'Padmaavat', 'genres': 'Action|Drama|History', 'keywords': 'queen rani history', 'tags': 'hindi historical epic'},
            {'movieId': 12, 'title': 'Dil Chahta Hai', 'genres': 'Comedy|Drama', 'keywords': 'friendship love maturity', 'tags': 'hindi cult classic'},
            {'movieId': 13, 'title': 'Omkara', 'genres': 'Crime|Drama|Thriller', 'keywords': 'mafia politics shakespeare', 'tags': 'hindi thriller crime'},
            {'movieId': 14, 'title': 'Devdas', 'genres': 'Drama|Musical|Romance', 'keywords': 'love tragedy alcoholism', 'tags': 'hindi romantic tragedy'},
            {'movieId': 15, 'title': 'Kabir Singh', 'genres': 'Action|Drama|Romance', 'keywords': 'doctor love obsession', 'tags': 'hindi romance drama'},
            {'movieId': 16, 'title': 'War', 'genres': 'Action|Thriller', 'keywords': 'spy undercover agent', 'tags': 'hindi action spy'},
            {'movieId': 17, 'title': 'Joker', 'genres': 'Drama|Thriller', 'keywords': 'mental asylum society', 'tags': 'hindi psychological thriller'},
            {'movieId': 18, 'title': 'Article 15', 'genres': 'Crime|Drama', 'keywords': 'caste discrimination police', 'tags': 'hindi social drama'},
            {'movieId': 19, 'title': 'Gully Boy', 'genres': 'Drama|Music', 'keywords': 'street rapper underground', 'tags': 'hindi music drama'},
            {'movieId': 20, 'title': 'Andhadhun', 'genres': 'Crime|Thriller', 'keywords': 'piano blind witness', 'tags': 'hindi thriller mystery'},
            {'movieId': 21, 'title': 'Tumbbad', 'genres': 'Drama|Fantasy|Horror', 'keywords': 'treasure curse mythology', 'tags': 'hindi horror fantasy'},
            {'movieId': 22, 'title': 'Badhaai Ho', 'genres': 'Comedy|Drama', 'keywords': 'pregnancy mature comedy', 'tags': 'hindi comedy family'},
            {'movieId': 23, 'title': 'Stree', 'genres': 'Comedy|Horror', 'keywords': 'ghost female legend', 'tags': 'hindi horror comedy'},
            {'movieId': 24, 'title': 'Raazi', 'genres': 'Action|Drama|Thriller', 'keywords': 'spy pakistan india', 'tags': 'hindi thriller spy'},
            {'movieId': 25, 'title': 'Sanju', 'genres': 'Biography|Drama', 'keywords': 'actor biography life', 'tags': 'hindi biographical drama'},
            {'movieId': 26, 'title': 'Baahubali: The Beginning', 'genres': 'Action|Adventure|Fantasy', 'keywords': 'kingdom warrior epic', 'tags': 'hindi epic fantasy'},
            {'movieId': 27, 'title': 'Baahubali 2: The Conclusion', 'genres': 'Action|Adventure|Fantasy', 'keywords': 'warrior king revenge', 'tags': 'hindi epic conclusion'},
            {'movieId': 28, 'title': 'Sairat', 'genres': 'Drama|Romance', 'keywords': 'caste love marriage', 'tags': 'hindi social drama'},
            {'movieId': 29, 'title': 'Uri: The Surgical Strike', 'genres': 'Action|Drama|Thriller', 'keywords': 'army surgical strike', 'tags': 'hindi military action'},
            {'movieId': 30, 'title': 'Chhichhore', 'genres': 'Comedy|Drama', 'keywords': 'college exam suicide', 'tags': 'hindi comedy drama'},
            
            # ==========================================
            # TAMIL MOVIES - ID 31-55
            # ==========================================
            {'movieId': 31, 'title': 'Vikram Vedha', 'genres': 'Action|Crime|Thriller', 'keywords': 'gangster police encounter', 'tags': 'tamil action thriller'},
            {'movieId': 32, 'title': 'Master', 'genres': 'Action|Drama', 'keywords': 'professor student gangster', 'tags': 'tamil action drama'},
            {'movieId': 33, 'title': 'Jai Bhim', 'genres': 'Action|Drama', 'keywords': 'tribal rights police', 'tags': 'tamil social drama'},
            {'movieId': 34, 'title': 'Soorarai Pottru', 'genres': 'Action|Drama', 'keywords': 'airline startup dream', 'tags': 'tamil inspirational'},
            {'movieId': 35, 'title': 'Asuran', 'genres': 'Action|Drama', 'keywords': 'farmer caste revenge', 'tags': 'tamil action drama'},
            {'movieId': 36, 'title': 'Kaithi', 'genres': 'Action|Crime', 'keywords': 'prisoner drug cartel', 'tags': 'tamil action crime'},
            {'movieId': 37, 'title': 'Anbe Sivam', 'genres': 'Adventure|Comedy|Drama', 'keywords': 'journey god man', 'tags': 'tamil inspirational'},
            {'movieId': 38, 'title': 'Vikram', 'genres': 'Action|Crime|Thriller', 'keywords': 'undercover agent gang', 'tags': 'tamil action spy'},
            {'movieId': 39, 'title': 'Ponniyin Selvan', 'genres': 'Action|Drama|History', 'keywords': 'king dynasty betrayal', 'tags': 'tamil historical epic'},
            {'movieId': 40, 'title': 'Jersey', 'genres': 'Drama|Sport', 'keywords': 'cricket comeback father', 'tags': 'tamil sports drama'},
            {'movieId': 41, 'title': 'Beast', 'genres': 'Action|Thriller', 'keywords': 'terrorist mall hostage', 'tags': 'tamil action thriller'},
            {'movieId': 42, 'title': 'Valimai', 'genres': 'Action|Drama', 'keywords': 'motorcycle gang police', 'tags': 'tamil action'},
            {'movieId': 43, 'title': 'Thiruchitrambalam', 'genres': 'Comedy|Drama|Romance', 'keywords': 'family love comedy', 'tags': 'tamil romantic comedy'},
            {'movieId': 44, 'title': 'Vada Chennai', 'genres': 'Crime|Drama', 'keywords': 'chennai mafia politics', 'tags': 'tamil crime drama'},
            {'movieId': 45, 'title': 'Karnan', 'genres': 'Action|Drama', 'keywords': 'caste violence revenge', 'tags': 'tamil action drama'},
            {'movieId': 46, 'title': 'Pariyerum Perumal', 'genres': 'Drama', 'keywords': 'caste discrimination college', 'tags': 'tamil social drama'},
            {'movieId': 47, 'title': 'Super Deluxe', 'genres': 'Drama|Mystery|Thriller', 'keywords': 'secrets family twisted', 'tags': 'tamil thriller'},
            {'movieId': 48, 'title': 'Mersal', 'genres': 'Action|Drama|Thriller', 'keywords': 'doctor twins revenge', 'tags': 'tamil action drama'},
            {'movieId': 49, 'title': 'Soodhu Kavvum', 'genres': 'Comedy|Crime', 'keywords': 'kidnap comedy', 'tags': 'tamil dark comedy'},
            {'movieId': 50, 'title': 'Visaranai', 'genres': 'Crime|Drama|Thriller', 'keywords': 'police custody torture', 'tags': 'tamil thriller'},
            {'movieId': 51, 'title': 'Mahanati', 'genres': 'Biography|Drama', 'keywords': 'actress biography legend', 'tags': 'tamil biographical'},
            {'movieId': 52, 'title': 'Kabali', 'genres': 'Action|Drama', 'keywords': 'gangster revenge love', 'tags': 'tamil action'},
            {'movieId': 53, 'title': '2.0', 'genres': 'Action|Sci-Fi', 'keywords': 'robot bird revenge', 'tags': 'tamil scifi action'},
            {'movieId': 54, 'title': 'Theri', 'genres': 'Action|Drama', 'keywords': 'police daughter love', 'tags': 'tamil action'},
            {'movieId': 55, 'title': 'Sivaji', 'genres': 'Action|Drama', 'keywords': 'social activist reform', 'tags': 'tamil action drama'},
            
            # ==========================================
            # TELUGU MOVIES - ID 56-80
            # ==========================================
            {'movieId': 56, 'title': 'RRR', 'genres': 'Action|Drama|History', 'keywords': 'revolution friendship british', 'tags': 'telugu historical epic'},
            {'movieId': 57, 'title': 'Pushpa', 'genres': 'Action|Drama', 'keywords': 'smuggler red sanders', 'tags': 'telugu action thriller'},
            {'movieId': 58, 'title': 'Baahubali: The Beginning', 'genres': 'Action|Adventure|Fantasy', 'keywords': 'kingdom warrior epic', 'tags': 'telugu epic fantasy'},
            {'movieId': 59, 'title': 'Baahubali 2: The Conclusion', 'genres': 'Action|Adventure|Fantasy', 'keywords': 'warrior king revenge', 'tags': 'telugu epic conclusion'},
            {'movieId': 60, 'title': 'Arjun Reddy', 'genres': 'Action|Drama|Romance', 'keywords': 'doctor love obsession', 'tags': 'telugu romance drama'},
            {'movieId': 61, 'title': 'Jersey', 'genres': 'Drama|Sport', 'keywords': 'cricket comeback father', 'tags': 'telugu sports'},
            {'movieId': 62, 'title': 'Ala Vaikunthapurramuloo', 'genres': 'Action|Drama', 'keywords': 'family switch love', 'tags': 'telugu action drama'},
            {'movieId': 63, 'title': 'Rangasthalam', 'genres': 'Action|Drama', 'keywords': 'village politics love', 'tags': 'telugu action'},
            {'movieId': 64, 'title': 'Fidaa', 'genres': 'Comedy|Drama|Romance', 'keywords': 'village love separation', 'tags': 'telugu romance'},
            {'movieId': 65, 'title': 'Maharshi', 'genres': 'Action|Drama', 'keywords': 'student teacher dreams', 'tags': 'telugu inspirational'},
            {'movieId': 66, 'title': 'Sye Raa', 'genres': 'Action|Drama|History', 'keywords': 'revolt british india', 'tags': 'telugu historical'},
            {'movieId': 67, 'title': 'Gopala Gopala', 'genres': 'Comedy|Drama', 'keywords': 'god contractor comedy', 'tags': 'telugu comedy'},
            {'movieId': 68, 'title': 'Janatha Garage', 'genres': 'Action|Drama', 'keywords': 'environment social', 'tags': 'telugu action drama'},
            {'movieId': 69, 'title': 'Bharat Ane Nenu', 'genres': 'Action|Drama', 'keywords': 'politics cm young', 'tags': 'telugu political'},
            {'movieId': 70, 'title': 'Radhe Shyam', 'genres': 'Drama|Romance', 'keywords': 'palmist love', 'tags': 'telugu romance'},
            {'movieId': 71, 'title': 'Akhanda', 'genres': 'Action|Drama', 'keywords': 'supernatural warrior', 'tags': 'telugu action'},
            {'movieId': 72, 'title': 'Shyam Singha Roy', 'genres': 'Drama|Fantasy', 'keywords': 'director ghost history', 'tags': 'telugu fantasy'},
            {'movieId': 73, 'title': 'Ala Modalaithe', 'genres': 'Comedy|Drama|Romance', 'keywords': 'love comedy family', 'tags': 'telugu romantic comedy'},
            {'movieId': 74, 'title': 'Eega', 'genres': 'Comedy|Drama|Fantasy', 'keywords': 'housefly reincarnation', 'tags': 'telugu fantasy comedy'},
            {'movieId': 75, 'title': 'Attarintiki Daredi', 'genres': 'Action|Comedy', 'keywords': 'family politics', 'tags': 'telugu action comedy'},
            {'movieId': 76, 'title': 'udu', 'genres': 'Action|Drama', 'keywords': 'blind gangster revenge', 'tags': 'telugu action thriller'},
            {'movieId': 77, 'title': 'Kshana Kshanam', 'genres': 'Action|Crime|Thriller', 'keywords': 'kidnap heist', 'tags': 'telugu thriller'},
            {'movieId': 78, 'title': 'Indra', 'genres': 'Action|Drama', 'keywords': 'smuggler politician', 'tags': 'telugu action'},
            {'movieId': 79, 'title': 'Okkadu', 'genres': 'Action|Drama', 'keywords': 'kabaddi player love', 'tags': 'telugu action sports'},
            {'movieId': 80, 'title': 'Pokiri', 'genres': 'Action|Drama', 'keywords': 'undercover cop gangster', 'tags': 'telugu action classic'},
            
            # ==========================================
            # KANNADA MOVIES - ID 81-105
            # ==========================================
            {'movieId': 81, 'title': 'KGF: Chapter 1', 'genres': 'Action|Crime|Drama', 'keywords': 'gold mine gangster', 'tags': 'kannada action crime'},
            {'movieId': 82, 'title': 'KGF: Chapter 2', 'genres': 'Action|Crime|Drama', 'keywords': 'gold empire revenge', 'tags': 'kannada action sequel'},
            {'movieId': 83, 'title': 'Kantara', 'genres': 'Action|Adventure|Drama', 'keywords': 'village myth creature', 'tags': 'kannada action adventure'},
            {'movieId': 84, 'title': '777 Charlie', 'genres': 'Adventure|Drama', 'keywords': 'dog adventure journey', 'tags': 'kannada adventure drama'},
            {'movieId': 85, 'title': 'Mahanal', 'genres': 'Drama|Mystery', 'keywords': 'orphanage mystery', 'tags': 'kannada mystery drama'},
            {'movieId': 86, 'title': 'Ulidavaru Kandanthe', 'genres': 'Crime|Drama|Thriller', 'keywords': 'journalist crime', 'tags': 'kannada crime thriller'},
            {'movieId': 87, 'title': 'Thithi', 'genres': 'Comedy|Drama', 'keywords': 'funeral family comedy', 'tags': 'kannada comedy drama'},
            {'movieId': 88, 'title': 'RangiTaranga', 'genres': 'Drama|Mystery|Romance', 'keywords': 'mystery love story', 'tags': 'kannada mystery romance'},
            {'movieId': 89, 'title': ' Lucia', 'genres': 'Drama|Mystery', 'keywords': 'memory loss cinema', 'tags': 'kannada psychological drama'},
            {'movieId': 90, 'title': 'Mungaru Male', 'genres': 'Drama|Romance', 'keywords': 'love journey mountains', 'tags': 'kannada romance classic'},
            {'movieId': 91, 'title': 'Yash', 'genres': 'Action|Drama', 'keywords': 'gangster revenge', 'tags': 'kannada action drama'},
            {'movieId': 92, 'title': 'Googly', 'genres': 'Action|Comedy', 'keywords': 'twins cricket', 'tags': 'kannada comedy action'},
            {'movieId': 93, 'title': 'Bell Bottom', 'genres': 'Action|Comedy', 'keywords': 'spy rescue comedy', 'tags': 'kannada action comedy'},
            {'movieId': 94, 'title': 'James', 'genres': 'Action|Drama', 'keywords': 'prisoner escape', 'tags': 'kannada action'},
            {'movieId': 95, 'title': 'Roberrt', 'genres': 'Action|Drama', 'keywords': 'construction worker hero', 'tags': 'kannada action'},
            {'movieId': 96, 'title': 'Salaga', 'genres': 'Action|Crime|Drama', 'keywords': 'undercover cop', 'tags': 'kannada action crime'},
            {'movieId': 97, 'title': 'Vikrant Rona', 'genres': 'Action|Adventure|Thriller', 'keywords': 'village mystery', 'tags': 'kannada action thriller'},
            {'movieId': 98, 'title': 'Sarkari Hi.5', 'genres': 'Comedy|Drama', 'keywords': 'government worker comedy', 'tags': 'kannada comedy'},
            {'movieId': 99, 'title': 'Myna', 'genres': 'Drama', 'keywords': 'orphan girl love', 'tags': 'kannada drama'},
            {'movieId': 100, 'title': 'Makkhi', 'genres': 'Action|Drama|Fantasy', 'keywords': 'fly revenge reincarnation', 'tags': 'kannada fantasy'},
            {'movieId': 101, 'title': 'Adyaksha', 'genres': 'Action|Comedy', 'keywords': 'doctor underworld', 'tags': 'kannada action comedy'},
            {'movieId': 102, 'title': 'Rajendrababu', 'genres': 'Action|Comedy', 'keywords': ' politician comedy', 'tags': 'kannada comedy'},
            {'movieId': 103, 'title': 'Shivaji', 'genres': 'Action|Drama', 'keywords': 'real estate social', 'tags': 'kannada action'},
            {'movieId': 104, 'title': 'Bajrangi', 'genres': 'Comedy|Drama', 'keywords': 'congress politics', 'tags': 'kannada comedy'},
            {'movieId': 105, 'title': 'Jigarthanda', 'genres': 'Action|Drama', 'keywords': 'filmmaker gangster', 'tags': 'kannada action drama'},
            
            # ==========================================
            # MALAYALAM MOVIES - ID 106-130
            # ==========================================
            {'movieId': 106, 'title': 'Drishyam', 'genres': 'Crime|Drama|Thriller', 'keywords': 'family crime hide', 'tags': 'malayalam thriller crime'},
            {'movieId': 107, 'title': 'Drishyam 2', 'genres': 'Crime|Drama|Thriller', 'keywords': 'investigation evidence', 'tags': 'malayalam thriller sequel'},
            {'movieId': 108, 'title': 'Lucifer', 'genres': 'Action|Crime|Drama', 'keywords': 'politician crime revenge', 'tags': 'malayalam action thriller'},
            {'movieId': 109, 'title': 'Pulimurugan', 'genres': 'Action|Adventure', 'keywords': 'tiger man hunter', 'tags': 'malayalam action adventure'},
            {'movieId': 110, 'title': 'Bangalore Days', 'genres': 'Drama|Romance', 'keywords': 'friendship love journey', 'tags': 'malayalam romance drama'},
            {'movieId': 111, 'title': 'Premam', 'genres': 'Comedy|Drama|Romance', 'keywords': 'love school college', 'tags': 'malayalam romantic comedy'},
            {'movieId': 112, 'title': 'Oppam', 'genres': 'Crime|Drama|Thriller', 'keywords': 'blind visually impaired', 'tags': 'malayalam thriller'},
            {'movieId': 113, 'title': 'Maheshinte Prathikaaram', 'genres': 'Drama', 'keywords': 'photographer revenge', 'tags': 'malayalam drama'},
            {'movieId': 114, 'title': 'Koode', 'genres': 'Drama', 'keywords': 'siblings reunion', 'tags': 'malayalam family drama'},
            {'movieId': 115, 'title': 'Kumbalangi Nights', 'genres': 'Drama', 'keywords': 'family dysfunction love', 'tags': 'malayalam family drama'},
            {'movieId': 116, 'title': 'Sudani from Nigeria', 'genres': 'Comedy|Drama', 'keywords': 'foreigner friendship', 'tags': 'malayalam comedy drama'},
            {'movieId': 117, 'title': 'Jallikattu', 'genres': 'Adventure|Drama', 'keywords': 'buffalo crowd chaos', 'tags': 'malayalam adventure'},
            {'movieId': 118, 'title': 'Virus', 'genres': 'Drama|Thriller', 'keywords': 'nipah virus outbreak', 'tags': 'malayalam thriller'},
            {'movieId': 119, 'title': 'Mahabali', 'genres': 'Comedy|Drama', 'keywords': 'government job comedy', 'tags': 'malayalam comedy'},
            {'movieId': 120, 'title': 'Take Off', 'genres': 'Drama|Thriller', 'keywords': 'kuwait evacuation', 'tags': 'malayalam thriller'},
            {'movieId': 121, 'title': 'Kammara Sambhavam', 'genres': 'Comedy|Drama|Fantasy', 'keywords': 'politician reincarnation', 'tags': 'malayalam comedy fantasy'},
            {'movieId': 122, 'title': 'Thondimuthalum Driksakshiyum', 'genres': 'Crime|Drama', 'keywords': 'police couple kidnap', 'tags': 'malayalam crime drama'},
            {'movieId': 123, 'title': 'Ath背诵i', 'genres': 'Drama|Mystery', 'keywords': 'missing mother', 'tags': 'malayalam mystery drama'},
            {'movieId': 124, 'title': 'Parava', 'genres': 'Drama', 'keywords': 'pigeon racing', 'tags': 'malayalam drama'},
            {'movieId': 125, 'title': 'Njan Steve Lopez', 'genres': 'Drama|Music', 'keywords': 'musician journey', 'tags': 'malayalam musical drama'},
            {'movieId': 126, 'title': 'Mollywood', 'genres': 'Comedy|Drama', 'keywords': 'cinema comedy', 'tags': 'malayalam comedy'},
            {'movieId': 127, 'title': 'Churuli', 'genres': 'Drama|Fantasy|Mystery', 'keywords': 'forest tunnel reality', 'tags': 'malayalam mystery fantasy'},
            {'movieId': 128, 'title': 'The Great Indian Kitchen', 'genres': 'Drama', 'keywords': 'housewife cooking', 'tags': 'malayalam slice of life'},
            {'movieId': 129, 'title': 'Ariyippu', 'genres': 'Drama', 'keywords': 'gulf return', 'tags': 'malayalam drama'},
            {'movieId': 130, 'title': 'Nanban', 'genres': 'Drama|Friendship', 'keywords': 'friends reunion', 'tags': 'malayalam drama friendship'},
            
            # ==========================================
            # HOLLYWOOD/INTERNATIONAL - ID 131-150
            # ==========================================
            {'movieId': 131, 'title': 'The Shawshank Redemption', 'genres': 'Drama|Crime', 'keywords': 'prison escape redemption', 'tags': 'hollywood classic'},
            {'movieId': 132, 'title': 'The Godfather', 'genres': 'Crime|Drama', 'keywords': 'mafia crime family', 'tags': 'hollywood classic'},
            {'movieId': 133, 'title': 'The Dark Knight', 'genres': 'Action|Crime|Drama', 'keywords': 'joker batman hero', 'tags': 'hollywood superhero'},
            {'movieId': 134, 'title': 'Pulp Fiction', 'genres': 'Crime|Drama', 'keywords': 'gangster nonlinear', 'tags': 'hollywood cult classic'},
            {'movieId': 135, 'title': 'Forrest Gump', 'genres': 'Drama|Romance', 'keywords': 'life journey love', 'tags': 'hollywood classic'},
            {'movieId': 136, 'title': 'Inception', 'genres': 'Action|Sci-Fi|Thriller', 'keywords': 'dream heist mind', 'tags': 'hollywood scifi'},
            {'movieId': 137, 'title': 'The Matrix', 'genres': 'Action|Sci-Fi', 'keywords': 'simulation reality', 'tags': 'hollywood scifi classic'},
            {'movieId': 138, 'title': 'Interstellar', 'genres': 'Adventure|Drama|Sci-Fi', 'keywords': 'space black hole', 'tags': 'hollywood scifi epic'},
            {'movieId': 139, 'title': 'Titanic', 'genres': 'Drama|Romance', 'keywords': 'ship love disaster', 'tags': 'hollywood romance'},
            {'movieId': 140, 'title': 'Avatar', 'genres': 'Action|Adventure|Sci-Fi', 'keywords': 'alien pandora', 'tags': 'hollywood scifi'},
            {'movieId': 141, 'title': 'Avengers: Endgame', 'genres': 'Action|Adventure|Sci-Fi', 'keywords': 'superhero team', 'tags': 'hollywood marvel'},
            {'movieId': 142, 'title': 'Joker', 'genres': 'Crime|Drama|Thriller', 'keywords': 'villain mental health', 'tags': 'hollywood DC'},
            {'movieId': 143, 'title': 'Parasite', 'genres': 'Comedy|Drama|Thriller', 'keywords': 'class tension', 'tags': 'korean cinema'},
            {'movieId': 144, 'title': 'Dune', 'genres': 'Action|Adventure|Sci-Fi', 'keywords': 'desert spice', 'tags': 'hollywood scifi'},
            {'movieId': 145, 'title': 'Spider-Man: No Way Home', 'genres': 'Action|Adventure|Sci-Fi', 'keywords': 'multiverse spider', 'tags': 'hollywood marvel'},
            {'movieId': 146, 'title': 'Everything Everywhere All at Once', 'genres': 'Action|Adventure|Comedy', 'keywords': 'multiverse quantum', 'tags': 'hollywood scifi comedy'},
            {'movieId': 147, 'title': 'Top Gun: Maverick', 'genres': 'Action|Drama', 'keywords': 'pilot fighter jet', 'tags': 'hollywood action'},
            {'movieId': 148, 'title': 'Barbie', 'genres': 'Comedy|Fantasy', 'keywords': 'doll barbie world', 'tags': 'hollywood comedy'},
            {'movieId': 149, 'title': 'Oppenheimer', 'genres': 'Biography|Drama|History', 'keywords': 'atomic bomb scientist', 'tags': 'hollywood historical'},
            {'movieId': 150, 'title': 'Mission: Impossible', 'genres': 'Action|Adventure|Thriller', 'keywords': 'spy mission impossible', 'tags': 'hollywood action'},
        ]
        
        movies_df = pd.DataFrame(movies_list)
        
        # Generate ratings with realistic patterns
        np.random.seed(42)
        num_users = 150  # More users
        num_movies = 150
        
        # Create user-movie ratings with realistic patterns
        ratings_list = []
        
        # Create user preferences based on language/genre affinity
        np.random.seed(42)
        
        # Language preferences: 0-29: Hindi, 30-54: Tamil, 55-79: Telugu, 80-104: Kannada, 105-129: Malayalam, 130-149: Hollywood
        # Users can have different affinities
        for user_id in range(1, num_users + 1):
            # Each user has different language preferences
            user_lang_prefs = np.random.dirichlet(np.ones(6))  # 6 language categories
            
            for movie_id in range(1, num_movies + 1):
                # Determine movie language category
                if movie_id <= 30:
                    lang_idx = 0  # Hindi
                elif movie_id <= 55:
                    lang_idx = 1  # Tamil
                elif movie_id <= 80:
                    lang_idx = 2  # Telugu
                elif movie_id <= 105:
                    lang_idx = 3  # Kannada
                elif movie_id <= 130:
                    lang_idx = 4  # Malayalam
                else:
                    lang_idx = 5  # Hollywood
                
                # Base rating influenced by language preference
                user_base = np.random.uniform(2.5, 4.5)
                lang_score = user_lang_prefs[lang_idx]
                
                # Base rating influenced by user preference + language affinity
                base_rating = user_base + 1.5 * lang_score
                
                # Add noise and clip to valid range
                rating = np.clip(base_rating + np.random.normal(0, 0.3), 1, 5)
                
                # 75% chance of rating (more ratings = better accuracy)
                if np.random.random() > 0.25:
                    ratings_list.append({
                        'userId': user_id,
                        'movieId': movie_id,
                        'rating': round(rating, 1)
                    })
        
        ratings_df = pd.DataFrame(ratings_list)
        
        print(f"Loaded {len(movies_df)} movies")
        print(f"  - Hindi (Bollywood): 30 movies")
        print(f"  - Tamil: 25 movies")
        print(f"  - Telugu: 25 movies")
        print(f"  - Kannada: 25 movies")
        print(f"  - Malayalam: 25 movies")
        print(f"  - Hollywood: 20 movies")
        print(f"Loaded {len(ratings_df)} ratings")
        print(f"Generated data for {num_users} users")
        
        return movies_df, ratings_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# =============================================================================
# SECTION 2: DATA PREPROCESSING
# =============================================================================

def preprocess_data(movies_df, ratings_df):
    """
    Preprocess the data:
    - Handle missing values
    - Normalize ratings
    - Merge movie metadata with ratings
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Handle missing values in movies
    print("\n1. Handling Missing Values...")
    print(f"   - Movies missing values: {movies_df.isnull().sum().sum()}")
    print(f"   - Ratings missing values: {ratings_df.isnull().sum().sum()}")
    
    # Fill missing values
    movies_df['genres'] = movies_df['genres'].fillna('')
    movies_df['keywords'] = movies_df['keywords'].fillna('')
    movies_df['tags'] = movies_df['tags'].fillna('')
    ratings_df = ratings_df.dropna()
    
    # Normalize ratings (min-max normalization to 0-1 range)
    print("\n2. Normalizing Ratings...")
    ratings_df['rating_normalized'] = (ratings_df['rating'] - ratings_df['rating'].min()) / \
                                       (ratings_df['rating'].max() - ratings_df['rating'].min())
    print(f"   - Rating range: [{ratings_df['rating'].min()}, {ratings_df['rating'].max()}]")
    print(f"   - Normalized range: [{ratings_df['rating_normalized'].min():.2f}, {ratings_df['rating_normalized'].max():.2f}]")
    
    # Merge movies with ratings
    print("\n3. Merging Datasets...")
    merged_df = ratings_df.merge(movies_df, on='movieId', how='left')
    print(f"   - Merged dataset shape: {merged_df.shape}")
    
    # Create combined text feature for content-based filtering
    print("\n4. Creating Combined Text Features...")
    movies_df['combined_features'] = movies_df['genres'].str.replace('|', ' ') + ' ' + \
                                       movies_df['keywords'] + ' ' + \
                                       movies_df['tags']
    print("   - Combined features created successfully")
    
    # Create user-item matrix
    print("\n5. Creating User-Item Matrix...")
    user_item_matrix = ratings_df.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating'
    ).fillna(0)
    print(f"   - User-Item matrix shape: {user_item_matrix.shape}")
    
    print("\nPreprocessing complete!")
    
    return movies_df, ratings_df, merged_df, user_item_matrix


# =============================================================================
# SECTION 3: CONTENT-BASED FILTERING
# =============================================================================

class ContentBasedFilter:
    """
    Content-Based Filtering using TF-IDF and Cosine Similarity.
    Uses movie genres, keywords, and tags to find similar movies.
    """
    
    def __init__(self, movies_df):
        """
        Initialize the Content-Based Filter.
        """
        self.movies_df = movies_df
        self.tfidf_matrix = None
        self.movie_similarity = None
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=3000,
            ngram_range=(1, 3)
        )
        
    def fit(self):
        """
        Fit the TF-IDF vectorizer and compute movie similarity matrix.
        """
        print("\n" + "=" * 60)
        print("CONTENT-BASED FILTERING MODEL")
        print("=" * 60)
        
        print("\n1. Applying TF-IDF Vectorization...")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies_df['combined_features']
        )
        print(f"   - TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        print("\n2. Computing Cosine Similarity...")
        self.movie_similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print(f"   - Similarity matrix shape: {self.movie_similarity.shape}")
        
        # Create movie index mapping
        self.movie_idx = pd.Series(
            self.movies_df.index, 
            index=self.movies_df['movieId']
        )
        
        print("\nContent-Based Model trained successfully!")
        
    def get_similar_movies(self, movie_id, top_n=10):
        """
        Get similar movies based on content.
        """
        if movie_id not in self.movie_idx:
            return pd.DataFrame()
        
        idx = self.movie_idx[movie_id]
        
        # Get similarity scores for this movie
        sim_scores = list(enumerate(self.movie_similarity[idx]))
        
        # Sort by similarity (descending), excluding the movie itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, score) for i, score in sim_scores if i != idx][:top_n]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Return similar movies
        result = self.movies_df.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
        result['similarity_score'] = similarity_scores
        
        return result


# =============================================================================
# SECTION 4: COLLABORATIVE FILTERING
# =============================================================================

class CollaborativeFilter:
    """
    Collaborative Filtering using User-Item matrix and cosine similarity.
    """
    
    def __init__(self, user_item_matrix, ratings_df):
        """
        Initialize the Collaborative Filter.
        """
        self.user_item_matrix = user_item_matrix.copy()
        self.ratings_df = ratings_df
        self.user_similarity = None
        self.item_similarity = None
        self.user_means = None
        self.item_means = None
        
        # Compute means for normalization
        self.user_means = self.user_item_matrix.replace(0, np.nan).mean(axis=1)
        self.user_means = self.user_means.fillna(self.user_means.mean())
        self.item_means = self.user_item_matrix.replace(0, np.nan).mean(axis=0)
        self.item_means = self.item_means.fillna(self.item_means.mean())
        
    def fit_user_based(self):
        """
        Compute user-user similarity matrix.
        """
        print("\n" + "=" * 60)
        print("COLLABORATIVE FILTERING (USER-BASED)")
        print("=" * 60)
        
        # Create normalized matrix
        normalized_matrix = self.user_item_matrix.copy()
        for user_id in normalized_matrix.index:
            user_ratings = normalized_matrix.loc[user_id]
            mask = user_ratings > 0
            if mask.any():
                user_mean = user_ratings[mask].mean()
                normalized_matrix.loc[user_id, mask] = user_ratings[mask] - user_mean
        
        print("\n1. Computing User Similarity Matrix...")
        self.user_similarity = cosine_similarity(normalized_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        print(f"   - User similarity shape: {self.user_similarity.shape}")
        
        print("\nUser-based Collaborative Model trained!")
        
    def fit_item_based(self):
        """
        Compute item-item similarity matrix.
        """
        print("\n" + "=" * 60)
        print("COLLABORATIVE FILTERING (ITEM-BASED)")
        print("=" * 60)
        
        print("\n1. Computing Item Similarity Matrix...")
        item_matrix = self.user_item_matrix.T
        self.item_similarity = cosine_similarity(item_matrix)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=item_matrix.index,
            columns=item_matrix.index
        )
        print(f"   - Item similarity shape: {self.item_similarity.shape}")
        
        print("\nItem-based Collaborative Model trained!")
        
    def predict_user_based(self, user_id, movie_id, k=20):
        """
        Predict rating using user-based collaborative filtering.
        """
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if movie_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.mean().mean()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        
        movie_ratings_series = self.user_item_matrix[movie_id]
        movie_raters = movie_ratings_series[movie_ratings_series > 0].index.tolist()
        
        if len(movie_raters) == 0:
            return self.user_means[user_id]
        
        similarities = self.user_similarity.loc[user_id, movie_raters]
        top_k = similarities.nlargest(k)
        
        if top_k.sum() == 0:
            return self.user_means[user_id]
        
        numerator = 0
        denominator = 0
        user_mean = self.user_means[user_id]
        
        for similar_user in top_k.index:
            sim = top_k[similar_user]
            rating = self.user_item_matrix.loc[similar_user, movie_id]
            user_mean_sim = self.user_means[similar_user]
            
            numerator += sim * (rating - user_mean_sim)
            denominator += abs(sim)
        
        if denominator == 0:
            return self.user_means[user_id]
        
        predicted_rating = user_mean + (numerator / denominator)
        
        return np.clip(predicted_rating, 1, 5)
    
    def predict_item_based(self, user_id, movie_id, k=20):
        """
        Predict rating using item-based collaborative filtering.
        """
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if movie_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.mean().mean()
        
        user_ratings = self.user_item_matrix.loc[user_id]
        
        rated_mask = user_ratings > 0
        rated_movies = user_ratings[rated_mask]
        
        if len(rated_movies) == 0:
            return self.item_means[movie_id]
        
        similarities = self.item_similarity.loc[movie_id, rated_movies.index]
        top_k = similarities.nlargest(k)
        
        if top_k.sum() == 0:
            return rated_movies.mean()
        
        numerator = 0
        denominator = 0
        movie_mean = self.item_means[movie_id]
        
        for similar_movie in top_k.index:
            sim = top_k[similar_movie]
            rating = rated_movies[similar_movie]
            item_mean = self.item_means[similar_movie]
            
            numerator += sim * (rating - item_mean)
            denominator += abs(sim)
        
        if denominator == 0:
            return rated_movies.mean()
        
        predicted_rating = movie_mean + (numerator / denominator)
        
        return np.clip(predicted_rating, 1, 5)
    
    def predict_for_user(self, user_id):
        """
        Predict ratings for all movies for a given user.
        """
        predictions = []
        
        for movie_id in self.user_item_matrix.columns:
            actual_rating = self.user_item_matrix.loc[user_id, movie_id]
            
            if actual_rating > 0:
                pred = actual_rating
            else:
                user_based_pred = self.predict_user_based(user_id, movie_id, k=20)
                item_based_pred = self.predict_item_based(user_id, movie_id, k=20)
                pred = 0.3 * user_based_pred + 0.7 * item_based_pred
            
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': pred,
                'actual_rating': actual_rating
            })
        
        return pd.DataFrame(predictions)


# =============================================================================
# SECTION 5: HYBRID RECOMMENDATION
# =============================================================================

class HybridRecommender:
    """
    Hybrid Recommendation System combining Content-Based and Collaborative Filtering.
    """
    
    def __init__(self, content_model, collab_model, movies_df):
        """
        Initialize the Hybrid Recommender.
        """
        self.content_model = content_model
        self.collab_model = collab_model
        self.movies_df = movies_df
        self.content_weight = 0.35
        self.collab_weight = 0.65
        
    def set_weights(self, content_weight, collab_weight):
        """
        Set the weights for combining predictions.
        """
        total = content_weight + collab_weight
        self.content_weight = content_weight / total
        self.collab_weight = collab_weight / total
        print(f"\nHybrid weights set: Content={self.content_weight:.2f}, Collab={self.collab_weight:.2f}")
        
    def recommend_for_user(self, user_id, top_n=10, exclude_rated=True):
        """
        Get top-N movie recommendations for a user.
        """
        print(f"\n{'='*60}")
        print(f"GENERATING RECOMMENDATIONS FOR USER {user_id}")
        print(f"{'='*60}")
        
        predictions = self.collab_model.predict_for_user(user_id)
        
        predictions = predictions.merge(
            self.movies_df[['movieId', 'title', 'genres']], 
            on='movieId'
        )
        
        print("\n1. Computing Content-Based Scores...")
        user_rated = self.collab_model.user_item_matrix.loc[user_id]
        rated_movies = user_rated[user_rated > 0].index.tolist()
        
        if len(rated_movies) > 0:
            content_scores = []
            for movie_id in predictions['movieId']:
                if movie_id in rated_movies:
                    content_scores.append(0)
                else:
                    sims = []
                    for rated_movie in rated_movies:
                        if rated_movie in self.content_model.movie_idx and \
                           movie_id in self.content_model.movie_idx:
                            idx1 = self.content_model.movie_idx[rated_movie]
                            idx2 = self.content_model.movie_idx[movie_id]
                            sim = self.content_model.movie_similarity[idx1, idx2]
                            sims.append(sim)
                    content_scores.append(np.mean(sims) if sims else 0)
            
            predictions['content_score'] = content_scores
        else:
            predictions['content_score'] = 0
        
        # Normalize scores
        max_content = predictions['content_score'].max()
        predictions['content_score'] = predictions['content_score'] / max_content if max_content > 0 else 0
        predictions['collab_score'] = (predictions['predicted_rating'] - 1) / 4
        
        # Calculate hybrid score
        predictions['hybrid_score'] = (
            self.content_weight * predictions['content_score'] + 
            self.collab_weight * predictions['collab_score']
        )
        
        if exclude_rated:
            predictions = predictions[predictions['actual_rating'] == 0]
        
        recommendations = predictions.sort_values('hybrid_score', ascending=False).head(top_n)
        
        print(f"\n2. Top {top_n} Recommendations Generated!")
        
        return recommendations[['movieId', 'title', 'genres', 'predicted_rating', 
                                'content_score', 'collab_score', 'hybrid_score']]
    
    def get_similar_movies(self, movie_id, top_n=10):
        """
        Get movies similar to a given movie using both approaches.
        """
        content_similar = self.content_model.get_similar_movies(movie_id, top_n)
        
        if movie_id in self.collab_model.item_similarity.index:
            collab_similar = self.collab_model.item_similarity.loc[movie_id].sort_values(ascending=False)
            collab_similar = collab_similar.head(top_n).reset_index()
            collab_similar.columns = ['movieId', 'collab_sim']
        else:
            collab_similar = pd.DataFrame(columns=['movieId', 'collab_sim'])
        
        result = content_similar.merge(collab_similar, on='movieId', how='outer').fillna(0)
        result['hybrid_sim'] = (self.content_weight * result['similarity_score'] + 
                                self.collab_weight * result['collab_sim'])
        result = result.sort_values('hybrid_sim', ascending=False).head(top_n)
        
        return result[['movieId', 'title', 'genres', 'similarity_score', 'collab_sim', 'hybrid_sim']]


# =============================================================================
# SECTION 6: MODEL EVALUATION
# =============================================================================

def evaluate_models(ratings_df, user_item_matrix, movies_df, test_size=0.2):
    """
    Evaluate the recommendation models.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=test_size, random_state=42
    )
    
    print(f"\n1. Data Split:")
    print(f"   - Training set: {len(train_ratings)} ratings")
    print(f"   - Test set: {len(test_ratings)} ratings")
    
    train_matrix = train_ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    print("\n2. Building Models on Training Data...")
    
    content_model = ContentBasedFilter(movies_df)
    content_model.fit()
    
    collab_model = CollaborativeFilter(train_matrix, train_ratings)
    collab_model.fit_user_based()
    collab_model.fit_item_based()
    
    print("\n3. Evaluating on Test Set...")
    predictions = []
    actuals = []
    
    for _, row in test_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual = row['rating']
        
        if user_id in train_matrix.index and movie_id in train_matrix.columns:
            user_pred = collab_model.predict_user_based(user_id, movie_id, k=20)
            item_pred = collab_model.predict_item_based(user_id, movie_id, k=20)
            pred = 0.3 * user_pred + 0.7 * item_pred
        else:
            pred = train_matrix.mean().mean()
        
        predictions.append(pred)
        actuals.append(actual)
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    correct_strict = sum(abs(np.array(actuals) - np.array(predictions)) <= 0.5)
    accuracy_strict = correct_strict / len(actuals) * 100
    
    correct_relaxed = sum(abs(np.array(actuals) - np.array(predictions)) <= 1.0)
    accuracy_relaxed = correct_relaxed / len(actuals) * 100
    
    correct_moderate = sum(abs(np.array(actuals) - np.array(predictions)) <= 0.75)
    accuracy_moderate = correct_moderate / len(actuals) * 100
    
    print(f"\n4. Evaluation Results:")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - Accuracy (within 0.5 star): {accuracy_strict:.2f}%")
    print(f"   - Accuracy (within 0.75 star): {accuracy_moderate:.2f}%")
    print(f"   - Accuracy (within 1.0 star): {accuracy_relaxed:.2f}%")
    
    print("\nEvaluation Complete!")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy_relaxed,
        'accuracy_strict': accuracy_strict,
        'accuracy_moderate': accuracy_moderate,
        'train_size': len(train_ratings),
        'test_size': len(test_ratings)
    }


# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the Hybrid Movie Recommendation System.
    """
    print("\n" + "=" * 60)
    print("HYBRID MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("Combining Content-Based & Collaborative Filtering")
    print("=" * 60)
    
    # Step 1: Load Data
    movies_df, ratings_df = load_or_generate_data()
    
    # Step 2: Preprocess Data
    movies_df, ratings_df, merged_df, user_item_matrix = preprocess_data(movies_df, ratings_df)
    
    # Step 3: Build Content-Based Model
    content_model = ContentBasedFilter(movies_df)
    content_model.fit()
    
    # Step 4: Build Collaborative Model
    collab_model = CollaborativeFilter(user_item_matrix, ratings_df)
    collab_model.fit_user_based()
    collab_model.fit_item_based()
    
    # Step 5: Create Hybrid Recommender
    hybrid_recommender = HybridRecommender(content_model, collab_model, movies_df)
    hybrid_recommender.set_weights(0.25, 0.75)
    
    # Step 6: Evaluate Models
    metrics = evaluate_models(ratings_df, user_item_matrix, movies_df)
    
    # Step 7: Generate Recommendations
    print("\n" + "=" * 60)
    print("GENERATING RECOMMENDATIONS")
    print("=" * 60)
    
    # User 1 - Bollywood recommendations
    print(f"\n--- Top 10 Movie Recommendations for User 1 (Hindi/Bollywood) ---")
    recommendations = hybrid_recommender.recommend_for_user(user_id=1, top_n=10, exclude_rated=True)
    print(recommendations.to_string(index=False))
    
    # User 5 - Tamil recommendations
    print(f"\n--- Top 10 Movie Recommendations for User 5 (Tamil) ---")
    recommendations = hybrid_recommender.recommend_for_user(user_id=5, top_n=10, exclude_rated=True)
    print(recommendations.to_string(index=False))
    
    # Similar movies
    print(f"\n--- Movies Similar to '3 Idiots' (movieId=2) ---")
    similar = hybrid_recommender.get_similar_movies(movie_id=2, top_n=10)
    print(similar.to_string(index=False))
    
    print(f"\n--- Movies Similar to 'Vikram' (movieId=38, Tamil) ---")
    similar = hybrid_recommender.get_similar_movies(movie_id=38, top_n=10)
    print(similar.to_string(index=False))
    
    print(f"\n--- Movies Similar to 'RRR' (movieId=56, Telugu) ---")
    similar = hybrid_recommender.get_similar_movies(movie_id=56, top_n=10)
    print(similar.to_string(index=False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SYSTEM SUMMARY")
    print("=" * 60)
    print(f"Movies in database: {len(movies_df)}")
    print(f"  - Hindi (Bollywood): 30")
    print(f"  - Tamil: 25")
    print(f"  - Telugu: 25")
    print(f"  - Kannada: 25")
    print(f"  - Malayalam: 25")
    print(f"  - Hollywood: 20")
    print(f"Total ratings: {len(ratings_df)}")
    print(f"Unique users: {ratings_df['userId'].nunique()}")
    print(f"Model Accuracy (within 1 star): {metrics['accuracy']:.2f}%")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print("\nHybrid Recommendation System Complete!")
    
    return hybrid_recommender, metrics


if __name__ == "__main__":
    hybrid_recommender, metrics = main()
