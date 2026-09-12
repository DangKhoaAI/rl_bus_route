//! Travel duration is sampled exactly once, when an edge is entered.

use crate::domain::Scenario;

pub fn edge_duration_s(
    scenario: &Scenario,
    route_id: u32,
    from_index: usize,
    entry_time_s: i64,
) -> i64 {
    let config = &scenario.config;
    let span = config.stops_per_route - 1;
    let edge_index =
        route_id as usize * span + from_index.min(config.stops_per_route.saturating_sub(2));
    let buckets = scenario.traffic_dims[1];
    let bucket = ((entry_time_s / config.traffic_bucket_s) as usize).min(buckets - 1);
    let seconds = scenario.network.edge_base_s[edge_index] as f64
        * scenario.traffic(edge_index, bucket) as f64;
    let tick = config.tick_s;
    let rounded = (seconds / tick as f64).ceil() as i64 * tick;
    rounded.max(tick)
}
