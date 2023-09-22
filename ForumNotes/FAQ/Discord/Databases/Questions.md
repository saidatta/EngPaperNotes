1. if you guys were doing a system design interview focused on relational db modeling, would you try to design the tables in a denormalized way to avoid joins during prod api calls? not sure if that's the best strategy, dont joins take a while? - https://discord.com/channels/993670645718204479/993670646330576990/1113541224368967721
	1. Broadly: Normalised data is more performant on write, because for each piece of state, there will be one and only one place to store it.
	   
	   It's less performant on read because you are constructing logical deductions out of a lattice of tiny pebbles of truth. It takes time to find, filter, sort and project those.
	   
	   Denormalisation "works" by skipping to a read-optimised schema for those applications where reads greatly outnumber writes. Which is very many.
	   
	   As an aside, "denormalised" doesn't have to mean "unstructured chaos". Do As Thou Wilt isn't necessarily the law of denormalising. There are well-studied, widely-used denormalisation techniques under the heading of "dimensional modelling".
	   
	   It's just that most of these discussions are dominated by web folk, not data warehouse folk.
	   
	   The shoutout to Event Sourcing is promising, given the interaction with the CQRS pattern. In short: read schema and write schema have different demand and performance characteristics. So split them up.