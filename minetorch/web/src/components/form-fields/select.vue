<template>
  <el-form-item :label="label">
    <el-select v-model="formData[name]" :placeholder="help" :multiple="multiple">
      <el-option v-for="item in options" :key="itemValue(item)" :label="itemLabel(item)" :value="itemValue(item)">
        <span>{{ itemLabel(item) }}</span>
      </el-option>
    </el-select>
  </el-form-item>
</template>
<script>
import formFieldMixin from 'components/form-fields/form-field'

export default {
  mixins: [formFieldMixin],
  props: {
    options: {
      type: Array,
      default: function() {
        return []
      }
    },
    multiple: {
      type: Boolean,
      default: false
    }
  },
  methods: {
    itemValue(item) {
      return item === Object(item) ? item.value : item
    },
    itemLabel(item) {
      return item === Object(item) ? item.label : item
    }
  }
}
</script>
<style lang="scss">
</style>
