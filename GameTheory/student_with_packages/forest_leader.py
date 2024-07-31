"""
A simple leader for the Stackelberg game.
"""
from base_leader import Leader
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
import numpy as np

___author___ = "Rokas"



class SimpleLeader(Leader):
    def __init__(self, name):
        # If you want to initialize something here, do it before the super() call.
        self.leader_prices = [] 
        self.follower_prices = []
        self.unit_cost = 1
        self.profits = []
        super().__init__(name)

    def calculate_daily_profit(self, leader_price):
        if self.model is not None:
            follower_price = self.model.predict([[leader_price]])[0]
        else:
            follower_price = 0  # Default value or some error handling
        demand = 2 - leader_price + 0.3 * follower_price
        profit = (leader_price - self.unit_cost) * demand
        self.profits.append(profit)  # Make sure to append profit after calculation
        return profit

    def new_price(self, date: int) -> float:
        self.log(date)
        if date == 101:
            self.analyse_history(days=date-1)
            best_price = self.leader_strategy()
            return float(best_price)
        elif date > 101:
            self.analyse_history(days=date-1)
            best_price = self.leader_strategy()
            return float(best_price)
        
    def evaluate_recent_performance(self):
        # Check if there are more than 5 profit entries
        if len(self.profits) > 5:
            recent_profits = self.profits[-5:]  # Get the last 5 entries for calculation
        else:
            # If fewer than 5 entries, use all available entries
            recent_profits = self.profits
        
        # Calculate the average profit if there are any profits recorded
        average_profit = sum(recent_profits) / len(recent_profits) if recent_profits else 0
        
        return average_profit


    def analyse_history(self, days=100, current_window_size=50):
        recent_performance = self.evaluate_recent_performance()
        performance_threshold = 0.5 # Set to zero for simplicity

        # Adjust window size based on performance
        if recent_performance < performance_threshold:
            new_window_size = max(20, current_window_size - 3)  # Decrease if underperforming
        else:
            new_window_size = min(100, current_window_size + 3)  # Increase if performing well

        self.leader_prices = []
        self.follower_prices = []
        for day in range(1, days + 1):
            leader_price, follower_price = self.get_price_from_date(day)
            self.leader_prices.append(leader_price)
            self.follower_prices.append(follower_price)

        start = max(0, days - new_window_size)
        X = np.array(self.leader_prices[start:]).reshape(-1, 1)
        y = np.array(self.follower_prices[start:])

        model = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_split=10, min_samples_leaf=4)
        model.fit(X, y)
        self.model = model  # Save the trained model
        return new_window_size  # Return the updated window size for next iteration


        

    def leader_strategy(self):
        upper_bound = 3.00
        potential_leader_prices = np.arange(1.00, upper_bound + 0.01, 0.01)
        max_profit = 0
        best_leader_price = 0
        for leader_price in potential_leader_prices:
            follower_price = self.model.predict([[leader_price]])[0]  # Use model to predict
            demand = 2 - leader_price + 0.3 * follower_price
            profit = (leader_price - self.unit_cost) * demand
            if profit > max_profit:
                max_profit = profit
                best_leader_price = leader_price
        return best_leader_price

    def start_simulation(self):
        self.log("Start of simulation")
        self.analyse_history()

    def end_simulation(self):
        """
        A function run at the end of the simulation.
        It calculates the total profit based on the leader and follower prices.
        """
        total_profit = 0
        unit_cost = 1
        all_leader_prices = [] 
        all_follower_prices = []
        for day in range(101, 131):  
            leader_price, follower_price = self.get_price_from_date(day)
            all_leader_prices.append(leader_price)
            all_follower_prices.append(follower_price)

        print(len(all_leader_prices))
        for i in range(len(all_leader_prices)):
            leader_price = all_leader_prices[i]
            follower_price = all_follower_prices[i]
            demand = 2 - leader_price + 0.3 * follower_price
            daily_profit = (leader_price - unit_cost) * demand
            total_profit += daily_profit
        
        self.log(f"End of simulation. Total profit: £{total_profit:.2f}")
        return total_profit



if __name__ == '__main__':
    # Make sure you set this to your group number!
    SimpleLeader('17Forest')
