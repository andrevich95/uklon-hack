from dataclasses import dataclass
from itertools import filterfalse
from datetime import datetime, timedelta
import pandas as pd

@dataclass
class Coordinates:
    lat: float
    lon: float

@dataclass
class Driver:
    id: int
    current_coord: Coordinates
    home_coord: Coordinates
    busy_till: datetime

@dataclass
class Order:
    id: int
    start_time: datetime
    pickup_coord: Coordinates
    dropoff_coord: Coordinates
    fare: float
    rating: int

@dataclass
class Ride:
    driver_id: int
    order_id: int
    distance_to_pickup: float
    distance_from_pickup_to_dropoff: float
    start_time: datetime
    end_time: datetime
    start_driver_coords: Coordinates

class SimulationEngine:
    def __init__(self, *, drivers_df, orders_df, simulation_step_delta=timedelta(minutes=10), available_orders_expire_time=timedelta(hours=2)):
        self._simulation_step_delta=simulation_step_delta
        self._available_orders_expire_time=available_orders_expire_time

        self._init_drivers(drivers_df)
        self._init_orders(orders_df)

        self._init_others()

    def _init_drivers(self, drivers_df):
        self._drivers_df = drivers_df

        self._drivers_list = []
        self._drivers_dict = {}

        for _, d in self._drivers_df.iterrows():
            coord = Coordinates(
                lat=d['lat'],
                lon=d['lon'],
            )

            driver = Driver(
                id=d['driverID'],
                home_coord=coord,
                current_coord=coord,
                busy_till=None,
            )

            self._drivers_list.append(
                driver
            )

            self._drivers_dict[driver.id] = driver

    def _init_orders(self, orders_df):
        self._orders_df = orders_df

        self._orders_df['pickupTime_converted'] = pd.to_datetime(self._orders_df['pickupTime'])

        self._orders_list = []
        self._orders_dict = {}

        for idx, o in self._orders_df.sort_values(by=['pickupTime']).iterrows():
            order = Order(
                id=idx,
                start_time=o['pickupTime_converted'],
                pickup_coord=Coordinates(
                    lat=o['pickup_lat'],
                    lon=o['pickup_lon'],
                ),
                dropoff_coord=Coordinates(
                    lat=o['dropoff_lat'],
                    lon=o['dropoff_lon'],
                ),
                fare=o['fare'],
                rating=o['rideRating'],
            )

            self._orders_list.append(order)
            self._orders_dict[order.id] = order

    def _init_others(self):
        _tmp_start_datetime = self._orders_list[0].start_time if self._orders_list[0] else datetime.now()
        _tmp_stop_datetime = self._orders_list[-1].start_time if self._orders_list[-1] else datetime.now()

        self._curr_simulation_time = _tmp_start_datetime.replace(hour=0, minute=0, second=0, microsecond=0) - self._simulation_step_delta
        self._stop_simulations_time = _tmp_stop_datetime + self._simulation_step_delta

        self._rides_history = []
        self._current_rides = []
        self._expired_orders = []
        self._available_orders = {}
        self._current_order_index = 0
        self._max_order_index = len(self._orders_list)

    def _finish_ride(self, ride):
        if ride.end_time < self._curr_simulation_time:
            self._drivers_dict[ride.driver_id].busy_till = None
            return True
        else:
            return False

    def _calculate_simulation_step(self):
        self._current_rides = list(filterfalse(self._finish_ride, self._current_rides))

        for order in self._available_orders.values():
            if order.start_time + self._available_orders_expire_time > self._curr_simulation_time:
                del self._available_orders[order.id]
                self._expired_orders.append(order)

        while self._current_order_index < self._max_order_index:
            c_order = self._orders_list[self._current_order_index]
            if c_order.start_time > self._curr_simulation_time:
                break
            self._available_orders[c_order.id] = c_order

            self._current_order_index += 1

    def __iter__(self):
        self._init_others()
        return self

    def __next__(self):
        self._curr_simulation_time = self._curr_simulation_time + self._simulation_step_delta
        self._calculate_simulation_step()
        
        if self._curr_simulation_time > self._stop_simulations_time and len(self._available_orders) == 0:
            raise StopIteration
        else:
            return self._curr_simulation_time

    def set_available_orders_expire_time(self, available_orders_expire_time=timedelta(hours=2)):
        self._available_orders_expire_time=available_orders_expire_time

    def set_simulation_step_delta(self, simulation_step_delta=timedelta(minutes=10)):
        self._simulation_step_delta=simulation_step_delta

    def get_available_drivers(self):
        return list(filter(lambda d: not d.busy_till, self._drivers_list))

    def get_available_orders(self):
        return list(self._available_orders.values())

    def get_rides_history(self):
        return self._rides_history
    
    def get_expired_order(self):
        return self._expired_orders

    def set_ride(self, *, driver_id, order_id, end_time, distance_to_pickup=0, distance_from_pickup_to_dropoff=0):
        if driver_id not in self._drivers_dict:
            raise ValueError('Provided driver not exist')
        if order_id not in self._orders_dict:
            raise ValueError('Provided order not exist')
        if order_id not in self._available_orders:
            raise ValueError('Provided order is not available')
        
        driver = self._drivers_dict[driver_id]

        if driver.busy_till:
            raise ValueError('Provided driver is not available')

        ride = Ride(
            driver_id=driver_id,
            order_id=order_id,
            distance_to_pickup=distance_to_pickup,
            distance_from_pickup_to_dropoff=distance_from_pickup_to_dropoff,
            start_time=self._curr_simulation_time,
            end_time=end_time,
            start_driver_coords=driver.current_coord,
        )

        driver.busy_till = end_time
        driver.current_coord = self._orders_dict[order_id].dropoff_coord

        self._rides_history.append(ride)
        self._current_rides.append(ride)
        del self._available_orders[order_id]

if __name__ == '__main__':
    orders = pd.read_csv('orders_ordered.csv')
    drivers = pd.read_csv('drivers.csv')

    se = SimulationEngine(
        drivers_df=drivers, 
        orders_df=orders,
        simulation_step_delta=timedelta(minutes=1),
    )

    se.set_simulation_step_delta(timedelta(minutes=10))
    se.set_available_orders_expire_time(timedelta(hours=1))

    # Baseline soluion
    for time in se:
        print(f'Simulation time {time}')
        orders = se.get_available_orders()
        drivers = se.get_available_drivers()

    #     print(f'Start sim step\n+++drivers amount: {len(drivers)}\n++Start orders amount: {len(orders)}')

        max_orders = len(orders)
        for i, driver in enumerate(drivers):
            if i < max_orders:
                se.set_ride(driver_id=driver.id, order_id=orders[i].id, end_time=time+timedelta(minutes=15), )
            else:
                break

    # for ride in se.get_rides_history():
    #     print(ride)
