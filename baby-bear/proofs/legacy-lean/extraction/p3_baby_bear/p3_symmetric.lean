import Hax

-- copied from p3_symmetric
namespace p3_symmetric.permutation

class Permutation.AssociatedTypes (Self : Type) (T : Type) where
  [trait_constr_Permutation_i0 : core_models.clone.Clone.AssociatedTypes Self]
  [trait_constr_Permutation_i1 : core_models.marker.Sync.AssociatedTypes Self]
  [trait_constr_Permutation_i2 : core_models.clone.Clone.AssociatedTypes T]

attribute [instance_reducible, instance]
  Permutation.AssociatedTypes.trait_constr_Permutation_i0

attribute [instance_reducible, instance]
  Permutation.AssociatedTypes.trait_constr_Permutation_i1

attribute [instance_reducible, instance]
  Permutation.AssociatedTypes.trait_constr_Permutation_i2

class Permutation (Self : Type) (T : Type)
  [associatedTypes : outParam (Permutation.AssociatedTypes (Self : Type) (T :
      Type))]
  where
  [trait_constr_Permutation_i0 : core_models.clone.Clone Self]
  [trait_constr_Permutation_i1 : core_models.marker.Sync Self]
  [trait_constr_Permutation_i2 : core_models.clone.Clone T]
  permute_mut (Self) (T) (self : Self) (input : T) : RustM T

attribute [instance_reducible, instance] Permutation.trait_constr_Permutation_i0

attribute [instance_reducible, instance] Permutation.trait_constr_Permutation_i1

attribute [instance_reducible, instance] Permutation.trait_constr_Permutation_i2

end p3_symmetric.permutation
