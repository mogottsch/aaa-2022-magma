# Bike Sharing Data Interpretation

Our data is extracted from the NextBike API.
We have a little bit of
[documentation](https://api.nextbike.net/api/documentation#nextbike_api),
however, a lot of columns are not explained and therefore we will write our
interpretations here.

## General

One entry in our dataset depicts the location of a bicycle in the system.
The system is free-floating, which means that bicycles can be rented and
returned anywhere in the service area.
However, there are still stations in the system, where bicycles can be returned
as well. We think that bicycles are relocated by the operator regularly to
these statons.

Variables that start with `p_` refer to a property of a place.  
Variables that start with `b_` refer to a property of a bike.  
See the
[documentation](https://api.nextbike.net/api/documentation#nextbike_api) for an
explanation of places.

| column                | dtype                  | null in %   | interpretation                                                                         |
| --------------------- | ---------------------- | ----------- | -------------------------------------------------------------------------------------- |
| `p_bike`              | `bool`                 | 0           | true if free-floating, false if at station                                             |
| `p_bikes`             | `int`                  | 0           | number of bikes at location (mostly = 1 for free-floating)                             |
| `p_lng`, `p_lat`      | `float`                | 0           | longitude and latitude of location                                                     |
| `b_number`            | `int`                  | 0           | id of the bike                                                                         |
| `p_spot`              | `bool`                 | 0           | almost the inverse of `p_bike`                                                         |
| `b_battery_pack`      | `json`                 | 99.98       | meta info (battery percentage) about battery for e bikes                               |
| `b_pedelec_battery`   | `float`                | 88.42       | battery percentage of pedelec                                                          |
| `p_address`           | `string`               | 98.09       | address of location                                                                    |
| `p_name`              | `string`               | 0           | name of station if at station else bike number/id                                      |
| `p_place_type`        | `int`                  | 0           | 12 for free-floating, 7 & 12 for stations                                              |
| `trip`                | `string`/`categorical` | 0           | star/end if location is start/end of trip, first/last means first/last position at day |
| --------------------- | ---------------------- | ----------- | -------------------------------------------------------------------------------------- |

### Columns with only one unique value

- `p_free_racks`
- `b_active`
- `p_bike_racks`
- `p_free_special_racks`
- `b_state`
- `p_rack_locks`
- `p_maintenance`
- `p_special_racks`
- `city`
