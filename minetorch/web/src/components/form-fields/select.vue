<template>
  <el-form-item :label="label">
    <el-select v-model="formData[name]" :placeholder="help" :multiple="multiple">
      <el-option v-for="item in options" :key="item.value" :label="item.label" :value="item.value">
        <span>{{ item.label }}</span>
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
  }
}
</script>
<style lang="scss">
</style>
