"""
Generates a synthetic-but-realistic dataset of citizen complaints for a
smart-city complaint routing system, modeled on real municipal 311/service-
request categories and phrasing patterns.

NOTE ON DATA PROVENANCE: real multi-category 311 complaint datasets (e.g.
NYC/Chicago Open Data) require API/domain access this environment doesn't
have. This generator produces varied, realistic complaint text using
templates + randomized details (streets, times, specifics) so the app has
a genuine multi-class dataset to train and evaluate on. See README for
details and a note on swapping in a real dataset later.
"""
import random
import pandas as pd

random.seed(42)

STREETS = ["Main St", "5th Avenue", "Oak Lane", "Riverside Drive", "Park Road",
           "Elm Street", "Church Road", "Station Road", "Highfield Ave",
           "Lakeview Drive", "Cedar Street", "Market Street", "Hillcrest Road",
           "Union Street", "Maple Avenue"]
TIMES = ["this morning", "last night", "for the past three days", "since Monday",
         "every evening this week", "over the weekend", "for two weeks now",
         "yesterday afternoon", "this past hour"]

TEMPLATES = {
    "Traffic & Roads": [
        "There's a large pothole on {street} that's been damaging cars {time}.",
        "The traffic light at the corner of {street} has been stuck on red {time}.",
        "Cars are parking illegally on {street} and blocking the bike lane {time}.",
        "The road on {street} is badly flooded {time}, it's not passable.",
        "Street signage near {street} is missing, causing confusion for drivers {time}.",
        "Traffic congestion on {street} has gotten much worse {time}, please review the signal timing.",
        "A stop sign near {street} was knocked down {time} and hasn't been replaced.",
        "The crosswalk markings on {street} have faded and are barely visible {time}.",
        "Speeding cars on {street} are a serious hazard {time}, we need a speed bump.",
        "The pavement is cracked and uneven on {street} {time}, cyclists keep swerving to avoid it.",
        "A pothole the size of a dinner plate has appeared on {street} {time}.",
        "Drivers are running the red light on {street} {time}, it's an accident waiting to happen.",
        "The road markings/lane lines on {street} have completely worn away {time}.",
        "There's a huge traffic jam on {street} {time} because of a broken-down vehicle.",
        "The pedestrian crossing button on {street} isn't working {time}.",
        "A broken guardrail on the bridge near {street} needs urgent road repair {time}.",
        "Roadworks on {street} have left an unmarked hole in the pavement {time}.",
        "The one-way sign on {street} is bent and hard to read {time}, causing wrong-way driving.",
        "Heavy trucks are damaging the road surface on {street} {time}.",
        "Please fix the sinkhole that opened up on {street} {time}, cars are swerving around it.",
    ],
    "Sanitation & Waste": [
        "Garbage collection on {street} was missed {time}, bins are overflowing.",
        "Someone has been illegally dumping construction waste near {street} {time}.",
        "The recycling bins on {street} haven't been emptied {time}.",
        "There's a bad smell coming from an uncollected dumpster on {street} {time}.",
        "Litter has been piling up along {street} {time}, it needs a cleanup crew.",
        "A dead animal has been left uncollected on {street} {time}.",
        "The public trash cans on {street} are overflowing {time} and attracting pests.",
        "Household waste has been dumped on the sidewalk of {street} {time}.",
        "Rats have been spotted near the uncollected trash on {street} {time}.",
        "The garbage truck skipped our block on {street} {time}, can someone come by?",
        "There's an old mattress and broken furniture dumped on {street} {time}.",
        "Yard waste hasn't been picked up on {street} {time} despite being scheduled.",
        "Flies are swarming around a garbage pile on {street} {time}.",
        "The compost bins provided on {street} are broken and leaking {time}.",
        "Someone burned trash illegally near {street} {time}, leaving ash everywhere.",
        "The street sweeper hasn't come through {street} {time}, leaves and debris are piling up.",
        "An overflowing skip/dumpster has been sitting on {street} {time} without being emptied.",
        "Recyclables are being mixed with regular trash by collectors on {street} {time}.",
    ],
    "Water & Utilities": [
        "There's a water main leak on {street} that's been flooding the street {time}.",
        "We've had no electricity on {street} {time}, please send a crew.",
        "Low water pressure has been reported on {street} {time}.",
        "A sewage smell is coming from a manhole on {street} {time}.",
        "The streetlights on {street} have been out {time}.",
        "There's a gas smell near {street} {time}, this may need urgent attention.",
        "Water discoloration from the tap has been reported by residents on {street} {time}.",
        "A burst pipe on {street} is spraying water onto the road {time}.",
        "The power keeps flickering on and off on {street} {time}.",
        "There's been a total blackout on {street} {time}, residents are asking when it'll be fixed.",
        "A fallen power line is dangling near {street} {time}, it looks dangerous.",
        "The water on {street} has been shut off {time} without any prior notice.",
        "A fire hydrant on {street} is leaking continuously {time}, wasting a lot of water.",
        "The gas meter box on {street} appears damaged {time}.",
        "Residents on {street} report the tap water smells strange {time}.",
        "An open manhole cover on {street} has been left uncovered {time}, it's a serious hazard.",
        "The electricity substation near {street} has been making loud buzzing noises {time}.",
    ],
    "Public Safety": [
        "A streetlamp on {street} has been flickering and going dark {time}, it feels unsafe at night.",
        "There's an abandoned building on {street} with a broken fence, unsafe {time}.",
        "Suspicious activity has been reported near {street} {time}, requesting a patrol.",
        "A fire hydrant on {street} appears damaged {time} and may not work in an emergency.",
        "The sidewalk on {street} is cracked and uneven {time}, a real trip hazard.",
        "A guard rail near {street} is broken {time} and needs urgent repair.",
        "Loose electrical wiring has been spotted hanging near {street} {time}.",
        "Someone has been loitering near the school on {street} {time}, parents are concerned.",
        "A security camera on {street} has been broken {time}, it's a blind spot now.",
        "There have been reports of break-ins near {street} {time}, residents want more patrols.",
        "An unsecured construction site on {street} has been left open {time}, kids could wander in.",
        "A dangerously leaning tree on {street} looks like it could fall {time}.",
        "Broken glass has been scattered on the pavement of {street} {time}.",
        "The emergency call box on {street} appears to be non-functional {time}.",
        "There's no working smoke detector reported in the shared stairwell on {street} {time}.",
        "A gate at the construction site on {street} has been left unlocked {time}.",
    ],
    "Noise Complaints": [
        "Loud construction work on {street} has been going on {time}, well outside permitted hours.",
        "A neighbor on {street} has been playing loud music {time}, disturbing the whole block.",
        "There's constant dog barking from a property on {street} {time}.",
        "A bar on {street} has had excessively loud events {time}.",
        "Car alarms keep going off on {street} {time} without anyone responding.",
        "Late-night parties on {street} {time} are keeping residents awake.",
        "Loud generator noise from a property on {street} has continued {time}.",
        "Motorbikes revving their engines on {street} {time} are extremely disruptive.",
        "A neighbor's renovation work on {street} started very early {time}, before permitted hours.",
        "Loudspeakers from an event on {street} can be heard blocks away {time}.",
        "Fireworks have been set off repeatedly near {street} {time}, disturbing residents.",
        "A restaurant's outdoor speakers on {street} are far too loud {time}.",
        "Continuous drilling noise from {street} {time} has been reported by several residents.",
        "A house party on {street} {time} has loud music going well past midnight.",
    ],
    "Parks & Public Spaces": [
        "The playground equipment at the park on {street} is broken {time}.",
        "Graffiti has appeared on the park walls near {street} {time}.",
        "The public restrooms at the park on {street} have been out of service {time}.",
        "Overgrown grass and weeds at the park near {street} need to be cut {time}.",
        "A bench in the park on {street} is damaged and unsafe to sit on {time}.",
        "Broken glass has been found near the playground on {street} {time}.",
        "The public water fountain in the park near {street} isn't working {time}.",
        "Fallen tree branches are blocking a path in the park on {street} {time}.",
        "The basketball court near {street} has large cracks in the surface {time}.",
        "Park lighting near {street} has stopped working {time}, it's dark and unsafe in the evenings.",
        "Litter bins in the park on {street} are missing or broken {time}.",
        "The swings at the playground on {street} are rusted and need replacing {time}.",
        "An unleashed aggressive dog has been reported at the park on {street} {time}.",
        "The community garden near {street} has been vandalized {time}.",
        "Park signage near {street} listing opening hours is faded and unreadable {time}.",
    ],
}


def generate_rows(n_per_category=150):
    rows = []
    for category, templates in TEMPLATES.items():
        for _ in range(n_per_category):
            t_idx = random.randrange(len(templates))
            template = templates[t_idx]
            text = template.format(street=random.choice(STREETS), time=random.choice(TIMES))
            rows.append({"text": text, "category": category, "template_id": f"{category}::{t_idx}"})
    return rows


if __name__ == "__main__":
    rows = generate_rows(n_per_category=150)
    df = pd.DataFrame(rows).drop_duplicates(subset="text").sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("data/complaints.csv", index=False)
    print(f"Generated {len(df)} unique complaint records")
    print(df["category"].value_counts())
