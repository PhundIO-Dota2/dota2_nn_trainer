package main

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/dotabuff/manta"
	"github.com/dotabuff/manta/dota"
	"log"
	"math"
	"os"
	"strings"
)

/* Represents the corpus of examples for one hero. */
type Corpus struct {
	MoveFile *os.File
	ItemFile *os.File
	Move     *bufio.Writer
	Item     *bufio.Writer

	ObservedItems           map[string]int
	ObservedAbilities       []string
	ObservedActiveAbilities map[string]int
	ObservedActiveItems     map[string]int
}

/* Represents a player to pay attention to in the first pass. */
type TopPlayer struct {
	Kills int32
	Name  string
}

type Hero struct {
	Team     uint64
	Entindex int32
}

const (
	TargetTower = iota + 1
	TargetBuilding
	TargetSelf
	TargetTree
	TargetJungle
	TargetLane
	TargetEnemyHero
	TargetFriendlyHero
)

/* Represents a move/attack example. */
type MoveExample struct {
	DotaTime   float32
	Health     float32
	Mana       float32
	CreepFront float32
	Level      float32
	CurrentX   float32
	CurrentY   float32

	OtherX [9]float32
	OtherY [9]float32

	AbilityCooldowns []float32
	CurrentItems     []int

	IsAttack float32
	MoveX    float32
	MoveY    float32

	Target      int
	AbilityUsed int
	ItemUsed    int
}

/* Writes a move example to CSV. */
func (example *MoveExample) WriteToCorpus(corpus *Corpus) {
	/* Input:
	   current time, health, mana, position of the creep front, XP level,
	   positions of all the players, ability cooldowns and current items.
	*/
	corpus.Move.WriteString(fmt.Sprintf("%f,%f,%f,%f,%f,%f,%f,",
		example.DotaTime,
		example.Health,
		example.Mana,
		example.Level,
		example.CreepFront,
		example.CurrentX,
		example.CurrentY,
	))

	for i := 0; i < 9; i++ {
		corpus.Move.WriteString(fmt.Sprintf("%f,%f,", example.OtherX[i], example.OtherY[i]))
	}

	for _, cooldown := range example.AbilityCooldowns {
		corpus.Move.WriteString(fmt.Sprintf("%f,", cooldown))
	}

	corpus.Move.WriteString("items,")

	for _, item := range example.CurrentItems {
		corpus.Move.WriteString(fmt.Sprintf("%d,", item))
	}

	/* Output:
	   move position, label of the target for abilities/attacks, label of the ability and/or item used
	*/
	corpus.Move.WriteString("output,")

	corpus.Move.WriteString(fmt.Sprintf("%f,%f,%f,%d,%d,%d\n",
		example.IsAttack,
		example.MoveX,
		example.MoveY,

		example.Target,
		example.AbilityUsed,
		example.ItemUsed,
	))
}

/* Represents an item/ability build example. */
type BuildExample struct {
}

/* Map and other game constants. */
const MIN_X = -8288.0
const MAX_X = 8288.0

const MIN_Y = -8288.0
const MAX_Y = 8288.0

const CELL_SIZE = 128.0

const COOLDOWN_SCALE = 360.0

const HANDLE_MAGIC = (1 << 14) - 1

/* Useful classnames. */
const TOWER = "CDOTA_BaseNPC_Tower"
const LANE_CREEP = "CDOTA_BaseNPC_Creep_Lane"
const JUNGLE_CREEP = "CDOTA_BaseNPC_Creep_Neutral"
const ANCIENT = "CDOTA_BaseNPC_Fort"
const RUNE = "CDOTA_Item_Rune"

/* Misc data. */
var teams []map[string]uint64 = []map[string]uint64{}
var corpora map[string][]*Corpus = make(map[string][]*Corpus)

/* Utility functions. */
func IsHero(ent *manta.PacketEntity) bool {
	return strings.HasPrefix(ent.ClassName, "CDOTA_Unit_Hero")
}

func IsItem(ent *manta.PacketEntity) bool {
	return strings.HasPrefix(ent.ClassName, "CDOTA_Item")
}

func IsAbility(ent *manta.PacketEntity) bool {
	return ent.ClassName == "CDOTABaseAbility" || strings.HasPrefix(ent.ClassName, "CDOTA_Ability") && ent.ClassName != "CDOTA_Ability_AttributeBonus"
}

/* Linearly maps coordinate components to [0, 1]. */
func RemapX(x float32) float32 {
	return (x+MIN_X)/(MAX_X-MIN_X) + 1
}

func RemapY(y float32) float32 {
	return (y+MIN_Y)/(MAX_Y-MIN_Y) + 1
}

/*
	Retrieves the location of an entity.

	Unlike the standard m_vecOrigin netprop in most Source games, Dota 2 splits an entity's location up into two parts in replays:

	- m_cellX, m_cellY: This represents which "cell" a location is in, with the map split into 128x128 cells.
	- m_offsetX, m_offsetY: This then represents the offset of an entity within the cell, relative to its lower left corner.

	(The Dota 2 map is 16577 x 16577 with the origin at its center as of 7.02. Most current resources for this kind of thing are for 6.xx, be wary!)
	This function takes those components and turns it into a regular Cartesian coordinate, since that's what the bot API uses.
*/
func GetLocation(ent *manta.PacketEntity) []float32 {
	cellX, _ := ent.FetchUint64("CBodyComponentBaseAnimatingOverlay.m_cellX")
	cellY, _ := ent.FetchUint64("CBodyComponentBaseAnimatingOverlay.m_cellY")

	offsetX, _ := ent.FetchFloat32("CBodyComponentBaseAnimatingOverlay.m_vecX")
	offsetY, _ := ent.FetchFloat32("CBodyComponentBaseAnimatingOverlay.m_vecY")

	return []float32{
		RemapX(float32(cellX)*CELL_SIZE - (MAX_X*2 + 1) + offsetX),
		RemapY(float32(cellY)*CELL_SIZE - (MAX_Y*2 + 1) + offsetY),
	}
}

/*
	Retrieves the Hammer name of an entity (name used by edicts, which Valve calls the classname).
	This is different from what Manta calls the "classname" (entity.ClassName), which is literally the name of the C++ class.
	(Isn't terminology just great).
*/
func GetHammerName(parser *manta.Parser, entity *manta.PacketEntity) string {
	if name_index, ok := entity.FetchInt32("CEntityIdentity.m_nameStringableIndex"); ok {
		if name, ok := parser.LookupStringByIndex("EntityNames", name_index); ok {
			return name
		}
	}

	return ""
}

/* Minimum index (for partial sorting players by kills in the first pass. No point in using a heap for just 3 elements) */
func MinIndex(top map[int32]*TopPlayer) int32 {
	best := int32(math.MaxInt32)
	bestIndex := int32(0)

	for i, player := range top {
		if player.Kills < best {
			best = player.Kills
			bestIndex = i
		}
	}

	return bestIndex
}

/* Opens a demo file. */
func OpenDemo(demo_name string) *os.File {
	filehandle, err := os.Open(demo_name)

	if err != nil {
		log.Fatal("Can't open demo")
	}

	return filehandle
}

/* Creates a Manta parser instance. */
func CreateParser(demo *os.File) *manta.Parser {
	parser, err := manta.NewStreamParser(demo)

	if err != nil {
		log.Fatalf("Unable to create parser: %s\n", err)
	}

	return parser
}

/* Returns or creates new corpus files for the given hero. */
func GetCorpus(hero string) []*Corpus {
	if corpus, ok := corpora[hero]; ok {
		return corpus
	} else {
		if err := os.Mkdir("data/"+hero, 493); err != nil && !os.IsExist(err) {
			log.Fatal("Can't create data folder")
		}

		radiant_move_file, radiant_move_err := os.Create("data/" + hero + "/2_moveexamples")
		radiant_items_file, radiant_item_err := os.Create("data/" + hero + "/2_itemsexamples")

		if radiant_move_err != nil || radiant_item_err != nil {
			log.Fatalf("Error creating corpus files for hero %s, team Radiant\n", hero)
		}

		dire_move_file, dire_move_err := os.Create("data/" + hero + "/3_moveexamples")
		dire_items_file, dire_item_err := os.Create("data/" + hero + "/3_itemsexamples")

		if dire_move_err != nil || dire_item_err != nil {
			log.Fatalf("Error creating corpus files for hero %s, team Dire\n", hero)
		}

		corpus := []*Corpus{
			{
				radiant_move_file,
				radiant_items_file,
				bufio.NewWriter(radiant_move_file),
				bufio.NewWriter(radiant_items_file),
				make(map[string]int),
				[]string{},
				make(map[string]int),
				make(map[string]int),
			},
			{
				dire_move_file,
				dire_items_file,
				bufio.NewWriter(dire_move_file),
				bufio.NewWriter(dire_items_file),
				make(map[string]int),
				[]string{},
				make(map[string]int),
				make(map[string]int),
			},
		}

		corpora[hero] = corpus
		return corpus
	}
}

/* Closes all the opened corpora files and writes the final ability/items/team composition data. */
func CloseCorpora() {
	/* Write ability_data.lua */
	activeAbilities := new(bytes.Buffer)
	activeItems := new(bytes.Buffer)
	items := new(bytes.Buffer)
	abilities := new(bytes.Buffer)

	activeAbilities.WriteString("activeAbilities = {") // start of table
	activeItems.WriteString("activeItems = {")
	items.WriteString("items = {")
	abilities.WriteString("abilities = {")

	for hero, corpus := range corpora {
		entry := fmt.Sprintf("%s={nil, {", hero) // map hero to team to abilities/items

		activeAbilities.WriteString(entry)
		activeItems.WriteString(entry)
		items.WriteString(entry)
		abilities.WriteString(entry)

		for _, team := range corpus {
			/* Add an entry for the id -> ability/item as well as ability/item -> id */
			for ability, id := range team.ObservedActiveAbilities {
				activeAbilities.WriteString(fmt.Sprintf("[%d]=\"%s\",%s=%d,", id, ability, ability, id))
			}

			for item, id := range team.ObservedActiveItems {
				activeItems.WriteString(fmt.Sprintf("[%d]=\"%s\",%s=%d,", id, item, item, id))
			}

			for item, id := range team.ObservedItems {
				items.WriteString(fmt.Sprintf("[%d]=\"%s\",%s=%d,", id, item, item, id))
			}

			for _, ability := range team.ObservedAbilities {
				abilities.WriteString(fmt.Sprintf("\"%s\",", ability))
			}

			activeAbilities.WriteString("},{") // close the table for that team
			activeItems.WriteString("},{")
			items.WriteString("},{")
			abilities.WriteString("},{")

			team.Move.Flush()
			team.Item.Flush()

			team.MoveFile.Close()
			team.ItemFile.Close()
		}

		activeAbilities.WriteString("}},") // close the table for that hero
		activeItems.WriteString("}},")
		items.WriteString("}},")
		abilities.WriteString("}},")
	}

	activeAbilities.WriteString("}\n")
	activeItems.WriteString("}\n")
	items.WriteString("}\n")
	abilities.WriteString("}\n")

	if observed_file, err := os.Create("ability_data.lua"); err == nil || os.IsExist(err) {
		writer := bufio.NewWriter(observed_file)

		defer observed_file.Close()
		defer writer.Flush()

		writer.WriteString("-- This is an automatically generated file. Do not modify.\n")
		writer.WriteString("module(\"ability_data\", package.seeall)\n")

		writer.WriteString(activeAbilities.String())
		writer.WriteString(activeItems.String())
		writer.WriteString(items.String())
		writer.WriteString(abilities.String())

		/* Also write team data (which isn't per corpus which is why we're doing it down here) */
		writer.WriteString("teams = {")

		for _, team := range teams {
			radiant := new(bytes.Buffer)
			dire := new(bytes.Buffer)

			for hero, team_num := range team {
				if team_num == 2 {
					radiant.WriteString(fmt.Sprintf("\"%s\",", hero))
				} else {
					dire.WriteString(fmt.Sprintf("\"%s\",", hero))
				}
			}

			writer.WriteString("{nil, {")
			writer.WriteString(radiant.String())
			writer.WriteString("},{")
			writer.WriteString(dire.String())
			writer.WriteString("}},")
		}

		writer.WriteString("}\n")
	} else {
		log.Fatalf("Error creating ability_data.lua")
	}
}

/*
	Retrieves the top 3 players on the winning team and also gets the start time of the match (horn) in ticks.
*/
func FirstPass(filehandle *os.File) (map[int32]*TopPlayer, uint32) {
	parser := CreateParser(filehandle)

	var startTime uint32
	var winningTeam int32

	top3 := make(map[int32]*TopPlayer)
	teamComposition := make(map[string]uint64)

	parser.OnPacketEntity(func(ent *manta.PacketEntity, _ manta.EntityEventType) error {
		if startTime == 0 && ent.ClassName == RUNE {
			startTime = parser.Tick
		} else if IsHero(ent) {
			name := GetHammerName(parser, ent)

			if _, ok := teamComposition[name]; !ok {
				if team, ok := ent.FetchUint64("m_iTeamNum"); ok {
					teamComposition[name] = team
				}
			}
		} else if ent.ClassName == ANCIENT {
			if health, ok := ent.FetchInt32("m_iHealth"); ok && health <= 0 { // ancient dead?
				if team, ok := ent.FetchUint64("m_iTeamNum"); ok {
					winningTeam = int32(team) ^ 1 // get the enemy team of the team whose ancient just died (2 ^ 1 == 3, 3 ^ 1 == 2)
				} else {
					log.Fatalf("Error retrieving m_iTeamNum from ancient (tick %d)\n", parser.Tick)
				}
			}
		} else if ent.ClassName == "CDOTA_PlayerResource" && winningTeam != 0 {
			for i := (winningTeam - 2) * 5; i < (winningTeam-1)*5; i++ {
				id := fmt.Sprintf("%04d", i)

				if kills, ok := ent.FetchInt32("m_vecPlayerTeamData." + id + ".m_iKills"); ok { // kill count
					if name, ok := ent.FetchString("m_vecPlayerData." + id + ".m_iszPlayerName"); ok { // name
						if len(top3) < 3 {
							top3[i] = &TopPlayer{kills, name}
						} else {
							min_index := MinIndex(top3)

							if min_player, _ := top3[min_index]; min_player.Kills < kills { // higher than lowest top 3 player, replace
								delete(top3, min_index)
								top3[i] = &TopPlayer{kills, name}
							}
						}
					}
				}
			}

			parser.Stop()
		}

		return nil
	})

	parser.Start()

	teams = append(teams, teamComposition)

	return top3, startTime
}

/*
	Tracks the actions of the top 3 players on the winning team and constructs examples out of each action.
*/
func SecondPass(filehandle *os.File, top3 map[int32]*TopPlayer, startTime float32) {
	parser := CreateParser(filehandle)

	heroes := make(map[string]*Hero)
	//creep_front := [2]float32{

	parser.OnPacketEntity(func(ent *manta.PacketEntity, _ manta.EntityEventType) error {
		if IsHero(ent) {
			if _, ok := heroes[ent.ClassName]; !ok {
				team, _ := ent.FetchUint64("m_iTeamNum")

				heroes[ent.ClassName] = &Hero{team, ent.Index}
			}
		}

		return nil
	})

	/* Callback for every unit action. */
	parser.Callbacks.OnCDOTAUserMsg_SpectatorPlayerUnitOrders(func(msg *dota.CDOTAUserMsg_SpectatorPlayerUnitOrders) error {
		if len(msg.GetUnits()) > 0 {
			for _, unit := range msg.GetUnits() { // multiple units can be selected
				entity := parser.PacketEntities[unit]

				if entity != nil {
					if IsHero(entity) { // replace with any criterion for producing examples
						id, ok := entity.FetchInt32("m_iPlayerID")

						if _, is_top3 := top3[id]; ok && is_top3 {
							/* Construct feature vector. */
							name := GetHammerName(parser, entity)

							team, _ := entity.FetchUint64("m_iTeamNum")
							corpus := GetCorpus(name)[team-2]
							ability_prefix := strings.SplitN(name, "dota_hero_", 2)[1]

							example := &MoveExample{}

							target := msg.GetTargetIndex()
							ability := msg.GetAbilityIndex()

							coords := GetLocation(entity)

							// Targeted ability or attack
							if target != 0 {
								example.IsAttack = 1.0

								if target == unit {
									example.Target = TargetSelf
									example.MoveX = coords[0]
									example.MoveY = coords[1]
								} else if target_ent, ok := parser.PacketEntities[target]; ok {
									target_coords := GetLocation(target_ent)
									example.MoveX = target_coords[0]
									example.MoveY = target_coords[1]

									if IsHero(target_ent) {
										target_team, _ := target_ent.FetchUint64("m_iTeamNum")

										if target_team == team {
											example.Target = TargetFriendlyHero
										} else {
											example.Target = TargetEnemyHero
										}
									} else {
										switch target_ent.ClassName {
										case LANE_CREEP:
											if ability == 0 {
												continue
											} else {
												example.Target = TargetLane
											}

										case JUNGLE_CREEP:
											example.Target = TargetJungle
										case TOWER:
											example.Target = TargetTower
										default:
											example.Target = TargetBuilding
										}
									}
								} else {
									// Manta corner case: packet entities are updated before callbacks so destroyed entities (eaten trees, picked up runes) are gone by this point
									//log.Fatalf("Error retrieving an attack target (entindex %d, tick %d, player %v)\n", target, parser.Tick, entity)
									example.Target = TargetTree // assume tree was eaten for now
									example.MoveX = coords[0]
									example.MoveY = coords[1]
								}
							}

							example.AbilityUsed = 1
							example.ItemUsed = 1

							// Ability used (not necessarily targeted)
							if ability != 0 {
								example.IsAttack = 1.0

								if target == 0 { // CWorld (self or location)
									example.Target = TargetSelf
									example.MoveX = coords[0]
									example.MoveY = coords[1]
								}

								if ability_ent, ok := parser.PacketEntities[ability]; ok {
									if IsItem(ability_ent) { // item
										if name := GetHammerName(parser, ability_ent); name != "" {
											if id, ok := corpus.ObservedActiveItems[name]; ok {
												example.ItemUsed = id + 1
											} else {
												/* New item, increase observed item count and assign ID */
												next_id := len(corpus.ObservedActiveItems) + 1

												corpus.ObservedActiveItems[name] = next_id
												example.ItemUsed = next_id + 1
											}
										}
									} else if IsAbility(ability_ent) { // ability
										if name := GetHammerName(parser, ability_ent); strings.HasPrefix(name, ability_prefix) {
											if id, ok := corpus.ObservedActiveAbilities[name]; ok {
												example.AbilityUsed = id + 1
											} else {
												/* New ability, increase observed ability count and assign ID */
												next_id := len(corpus.ObservedActiveAbilities) + 1

												corpus.ObservedActiveAbilities[name] = next_id
												example.AbilityUsed = next_id + 1
											}
										}
									}
								} else {
									// Could be caused by leveling an ability? Silence for now
									//log.Fatalf("Error retrieving an ability (entindex %d, tick %d, player %v)\n", ability, parser.Tick, entity)
									continue
								}
							}

							health, _ := entity.FetchInt32("m_iHealth")
							maxHealth, _ := entity.FetchInt32("m_iMaxHealth")
							mana, _ := entity.FetchFloat32("m_flMana")
							maxMana, _ := entity.FetchFloat32("m_flMaxMana")
							level, _ := entity.FetchInt32("m_iCurrentLevel")

							move_pos := msg.GetPosition()

							example.DotaTime = (float32(parser.Tick) - startTime) / 108000.0 // DotaTime()
							example.Health = float32(health) / float32(maxHealth)            // :GetHealth()
							example.Mana = mana / maxMana                                    // :GetMana()
							example.Level = float32(level) / 25.0                            // :GetCurrentLevel()
							example.CreepFront = 0.0                                         // GetLaneFrontAmount() FIXME

							// my position
							example.CurrentX = coords[0]
							example.CurrentY = coords[1]

							// everyone else's position
							ally := 0
							enemy := 4

							for _, hero := range heroes {
								if hero.Entindex == entity.Index {
									continue
								}

								loc := GetLocation(parser.PacketEntities[hero.Entindex])

								if hero.Team == team {
									example.OtherX[ally] = loc[0]
									example.OtherY[ally] = loc[1]
									ally++
								} else {
									example.OtherX[enemy] = loc[0]
									example.OtherY[enemy] = loc[1]
									enemy++
								}
							}

							// Retrieve ability cooldowns
							ability_id := 0
							for ability_count := 0; ; ability_count++ {
								if ability_handle, ok := entity.FetchUint32(fmt.Sprintf("m_hAbilities.%04d", ability_count)); ok {
									if ability, ok := parser.PacketEntities[int32(ability_handle&HANDLE_MAGIC)]; ok {
										if name := GetHammerName(parser, ability); strings.HasPrefix(name, ability_prefix) {
											if level, ok := ability.FetchInt32("m_iLevel"); level == 0 || !ok {
												example.AbilityCooldowns = append(example.AbilityCooldowns, 1.0)
											} else if cooldown, ok := ability.FetchFloat32("m_fCooldown"); ok {
												example.AbilityCooldowns = append(example.AbilityCooldowns, cooldown/360)
											}

											if len(corpus.ObservedAbilities) <= ability_id {
												corpus.ObservedAbilities = append(corpus.ObservedAbilities, name)
											}

											ability_id++
										}
									}
								} else {
									break
								}
							}

							// Retrieve current items
							for item_count := 0; ; item_count++ {
								if item_handle, ok := entity.FetchUint32(fmt.Sprintf("m_hItems.%04d", item_count)); ok {
									if item, ok := parser.PacketEntities[int32(item_handle&HANDLE_MAGIC)]; ok {
										if name := GetHammerName(parser, item); name != "" {
											if id, ok := corpus.ObservedItems[name]; ok {
												example.CurrentItems = append(example.CurrentItems, id)
											} else {
												next_id := len(corpus.ObservedItems) + 1

												corpus.ObservedItems[name] = next_id
												example.CurrentItems = append(example.CurrentItems, next_id)
											}
										}
									}
								} else {
									break
								}
							}

							if move_pos != nil {
								example.MoveX = RemapX(move_pos.GetX())
								example.MoveY = RemapY(move_pos.GetY())
							}

							example.WriteToCorpus(corpus)
						}
					}
				}
			}
		}

		return nil
	})

	parser.Start()
}

func main() {
	defer CloseCorpora()

	log.SetOutput(os.Stdout)

	if err := os.Mkdir("data", 493); err != nil && !os.IsExist(err) {
		log.Fatal("Can't create data folder")
	}

	if len(os.Args) == 1 {
		log.Fatal("usage: corpus_build <demos...>")
	}

	for i, demo_name := range os.Args[1:] {
		log.Printf("Demo %d (%s)\n", i+1, demo_name)

		filehandle := OpenDemo(demo_name)
		defer filehandle.Close()

		top3, startTime := FirstPass(filehandle) // retrieve top 3 players

		for id, player := range top3 {
			log.Println(id, player.Name, player.Kills)
		}

		filehandle.Seek(0, 0) // go back to beginning of demo

		SecondPass(filehandle, top3, float32(startTime)) // make examples
	}
}
